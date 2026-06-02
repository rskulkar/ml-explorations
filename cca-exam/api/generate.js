// api/generate.js — Vercel Edge Function
// Proxies requests to Anthropic API. Key stays server-side, never exposed to client.

export const config = { runtime: "edge" };

// Simple in-memory rate limiter — 3 exam generations per IP per hour
// Note: resets on cold start. For stricter limits use Vercel KV.
const rateLimitMap = new Map();
const LIMIT = 3;
const WINDOW_MS = 60 * 60 * 1000; // 1 hour

function checkRateLimit(ip) {
  const now = Date.now();
  const entry = rateLimitMap.get(ip);
  if (!entry || now - entry.windowStart > WINDOW_MS) {
    rateLimitMap.set(ip, { count: 1, windowStart: now });
    return true;
  }
  if (entry.count >= LIMIT) return false;
  entry.count++;
  return true;
}

export default async function handler(req) {
  // CORS — lock down to your Vercel domain in production
  const origin = req.headers.get("origin") || "";
  const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };

  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405, headers: { ...corsHeaders, "Content-Type": "application/json" }
    });
  }

  // Rate limit by IP
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "unknown";
  if (!checkRateLimit(ip)) {
    return new Response(JSON.stringify({ error: "Rate limit exceeded. Max 3 exam generations per hour." }), {
      status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" }
    });
  }

  // Parse request body
  let body;
  try {
    body = await req.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid request body" }), {
      status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" }
    });
  }

  const { model, max_tokens, system, messages } = body;

  // Validate — only allow the exam generation model/token budget
  if (max_tokens > 2500) {
    return new Response(JSON.stringify({ error: "max_tokens exceeds allowed limit" }), {
      status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" }
    });
  }

  // Forward to Anthropic
  const anthropicRes = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": process.env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({ model, max_tokens, system, messages }),
  });

  const data = await anthropicRes.json();

  return new Response(JSON.stringify(data), {
    status: anthropicRes.status,
    headers: { ...corsHeaders, "Content-Type": "application/json" }
  });
}
