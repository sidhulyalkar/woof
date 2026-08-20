import { NextResponse } from "next/server"
import { mockLeaderboard } from "@/lib/mock-data"

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const period = searchParams.get("period") || "weekly"

  await new Promise((resolve) => setTimeout(resolve, 400))

  return NextResponse.json({
    success: true,
    data: mockLeaderboard,
    period,
  })
}
