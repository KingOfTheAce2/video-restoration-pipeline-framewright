#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const BASE_URL = process.env.VLLM_BASE_URL || "http://localhost:8000/v1";
const MODEL = process.env.VLLM_MODEL || "Qwen/Qwen2.5-Coder-7B-Instruct";
const API_KEY = process.env.VLLM_API_KEY || "not-needed";

const server = new McpServer({
  name: "qwen-coder",
  version: "1.0.0",
});

server.tool(
  "qwen_code",
  "Generate, refactor, or transform code using the local Qwen 3 Coder model. " +
  "Provide a task description and optional code context. Returns generated code.",
  {
    task: z.string().describe("What to do: e.g. 'Write a Python function that...' or 'Refactor this code to...'"),
    code_context: z.string().optional().describe("Existing code to provide as context for the task"),
    language: z.string().optional().describe("Target programming language (e.g. python, typescript)"),
    max_tokens: z.number().optional().default(4096).describe("Maximum tokens in the response"),
  },
  async ({ task, code_context, language, max_tokens }) => {
    const messages = [
      {
        role: "system",
        content: "You are an expert coding assistant. Respond with clean, well-structured code. " +
                 "Include brief comments but minimize prose. If the task is ambiguous, make reasonable assumptions.",
      },
    ];

    let userContent = task;
    if (language) userContent += `\n\nTarget language: ${language}`;
    if (code_context) userContent += `\n\nExisting code context:\n\`\`\`\n${code_context}\n\`\`\``;

    messages.push({ role: "user", content: userContent });

    const response = await fetch(`${BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${API_KEY}`,
      },
      body: JSON.stringify({
        model: MODEL,
        messages,
        max_tokens: max_tokens || 4096,
        temperature: 0.2,
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      return { content: [{ type: "text", text: `Error from vLLM: ${response.status} ${err}` }] };
    }

    const data = await response.json();
    const text = data.choices?.[0]?.message?.content || "No response generated.";
    return { content: [{ type: "text", text }] };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
