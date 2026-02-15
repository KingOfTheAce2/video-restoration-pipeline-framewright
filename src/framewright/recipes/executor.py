"""Recipe executor for running restoration workflows."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .library import Recipe, RecipeStep

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of executing a recipe step."""
    step: RecipeStep
    success: bool
    skipped: bool = False
    error: Optional[str] = None
    duration_seconds: float = 0.0
    output_path: Optional[Path] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecipeResult:
    """Result of executing a complete recipe."""
    recipe: Recipe
    success: bool
    step_results: List[StepResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    output_path: Optional[Path] = None
    error: Optional[str] = None

    @property
    def completed_steps(self) -> int:
        return sum(1 for r in self.step_results if r.success and not r.skipped)

    @property
    def skipped_steps(self) -> int:
        return sum(1 for r in self.step_results if r.skipped)

    @property
    def failed_steps(self) -> int:
        return sum(1 for r in self.step_results if not r.success and not r.skipped)


class RecipeExecutor:
    """Executes restoration recipes step by step."""

    def __init__(
        self,
        processors: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ):
        """Initialize executor.

        Args:
            processors: Dict mapping processor names to processor instances
            dry_run: If True, don't actually run processors
        """
        self._processors = processors or {}
        self.dry_run = dry_run
        self._context: Dict[str, Any] = {}

    def register_processor(self, name: str, processor: Any) -> None:
        """Register a processor."""
        self._processors[name] = processor

    def execute(
        self,
        recipe: Recipe,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        step_callback: Optional[Callable[[StepResult], None]] = None,
    ) -> RecipeResult:
        """Execute a recipe.

        Args:
            recipe: Recipe to execute
            input_path: Input video path
            output_path: Output video path
            progress_callback: Called with (current_step, total_steps, step_name)
            step_callback: Called after each step completes

        Returns:
            Recipe execution result
        """
        import time

        result = RecipeResult(recipe=recipe, success=True)
        start_time = time.time()

        # Initialize context
        self._context = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "current_path": str(input_path),  # Tracks intermediate outputs
        }

        total_steps = len(recipe.steps)
        current_input = input_path

        logger.info(f"Executing recipe: {recipe.title}")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")

        for i, step in enumerate(recipe.steps):
            if progress_callback:
                progress_callback(i + 1, total_steps, step.name)

            # Check skip condition
            if step.skip_condition:
                try:
                    should_skip = eval(step.skip_condition, {"context": self._context})
                    if should_skip:
                        logger.info(f"Skipping step {step.order}: {step.name} (condition met)")
                        step_result = StepResult(step=step, success=True, skipped=True)
                        result.step_results.append(step_result)
                        if step_callback:
                            step_callback(step_result)
                        continue
                except Exception as e:
                    logger.warning(f"Failed to evaluate skip condition: {e}")

            # Execute step
            step_result = self._execute_step(step, current_input, i == total_steps - 1, output_path)
            result.step_results.append(step_result)

            if step_callback:
                step_callback(step_result)

            if not step_result.success and not step.optional:
                result.success = False
                result.error = f"Step {step.order} failed: {step_result.error}"
                logger.error(result.error)
                break

            # Update current input for next step
            if step_result.output_path:
                current_input = step_result.output_path
                self._context["current_path"] = str(current_input)

        result.total_duration_seconds = time.time() - start_time
        result.output_path = output_path if result.success else None

        logger.info(f"Recipe completed: success={result.success}, "
                   f"duration={result.total_duration_seconds:.1f}s")

        return result

    def _execute_step(
        self,
        step: RecipeStep,
        input_path: Path,
        is_final: bool,
        final_output: Path,
    ) -> StepResult:
        """Execute a single recipe step."""
        import time
        import tempfile

        start_time = time.time()

        logger.info(f"Executing step {step.order}: {step.name}")
        logger.debug(f"  Processor: {step.processor}")
        logger.debug(f"  Config: {step.config}")

        if self.dry_run:
            logger.info(f"  [DRY RUN] Would execute {step.processor}")
            return StepResult(
                step=step,
                success=True,
                duration_seconds=0.0,
                output_path=input_path,
            )

        # Get processor
        processor = self._processors.get(step.processor)
        if processor is None:
            error = f"Processor not found: {step.processor}"
            logger.error(error)
            return StepResult(step=step, success=False, error=error)

        # Determine output path
        if is_final:
            output_path = final_output
        else:
            # Create temporary output for intermediate steps
            suffix = Path(input_path).suffix
            output_path = Path(tempfile.mktemp(suffix=suffix))

        try:
            # Execute processor
            if hasattr(processor, "process"):
                # Standard processor interface
                result = processor.process(
                    input_path=input_path,
                    output_path=output_path,
                    config=step.config,
                    context=self._context,
                )
            elif callable(processor):
                # Callable processor
                result = processor(
                    input_path,
                    output_path,
                    step.config,
                    self._context,
                )
            else:
                raise ValueError(f"Unknown processor type: {type(processor)}")

            # Update context with result
            if isinstance(result, dict):
                self._context.update(result)

            duration = time.time() - start_time
            logger.info(f"  Completed in {duration:.1f}s")

            return StepResult(
                step=step,
                success=True,
                duration_seconds=duration,
                output_path=output_path,
                metrics=result if isinstance(result, dict) else {},
            )

        except Exception as e:
            duration = time.time() - start_time
            error = str(e)
            logger.error(f"  Failed: {error}")

            return StepResult(
                step=step,
                success=False,
                error=error,
                duration_seconds=duration,
            )

    def validate_recipe(self, recipe: Recipe) -> List[str]:
        """Validate that all processors are available.

        Returns:
            List of missing processor names
        """
        missing = []
        for step in recipe.steps:
            if step.processor not in self._processors:
                missing.append(step.processor)
        return missing

    def estimate_time(self, recipe: Recipe, video_duration: float) -> float:
        """Estimate execution time.

        Args:
            recipe: Recipe to estimate
            video_duration: Video duration in seconds

        Returns:
            Estimated time in seconds
        """
        # Parse estimated_time like "2x realtime"
        est = recipe.estimated_time.lower()
        if "realtime" in est:
            try:
                multiplier = float(est.split("x")[0])
                return video_duration * multiplier
            except:
                pass
        return video_duration * 2  # Default estimate

    def print_plan(self, recipe: Recipe) -> str:
        """Print execution plan for a recipe."""
        lines = [
            f"Execution Plan: {recipe.title}",
            "=" * 50,
        ]

        for step in recipe.steps:
            status = "available" if step.processor in self._processors else "MISSING"
            optional = " (optional)" if step.optional else ""
            lines.append(f"{step.order}. {step.name}{optional}")
            lines.append(f"   Processor: {step.processor} [{status}]")
            if step.notes:
                lines.append(f"   Note: {step.notes}")

        missing = self.validate_recipe(recipe)
        if missing:
            lines.extend(["", "WARNING: Missing processors:", *[f"  - {p}" for p in missing]])

        return "\n".join(lines)
