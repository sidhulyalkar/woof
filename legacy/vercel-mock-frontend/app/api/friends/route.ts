import { NextResponse } from "next/server"
import { mockFriends, mockFriendRequests } from "@/lib/mock-data"

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const type = searchParams.get("type") || "all"

  await new Promise((resolve) => setTimeout(resolve, 300))

  if (type === "requests") {
    return NextResponse.json({
      success: true,
      data: mockFriendRequests,
    })
  }

  return NextResponse.json({
    success: true,
    data: mockFriends,
  })
}

export async function POST(request: Request) {
  const body = await request.json()

  await new Promise((resolve) => setTimeout(resolve, 400))

  return NextResponse.json({
    success: true,
    message: "Friend request sent",
  })
}
