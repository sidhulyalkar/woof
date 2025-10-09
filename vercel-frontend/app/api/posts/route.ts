import { NextResponse } from "next/server"
import { mockPosts } from "@/lib/mock-data"

export async function GET() {
  await new Promise((resolve) => setTimeout(resolve, 400))

  return NextResponse.json({
    success: true,
    data: mockPosts,
  })
}

export async function POST(request: Request) {
  const body = await request.json()

  await new Promise((resolve) => setTimeout(resolve, 500))

  const newPost = {
    id: `post${Date.now()}`,
    ...body,
    timestamp: new Date().toISOString(),
    likes: 0,
    comments: 0,
    isLiked: false,
  }

  return NextResponse.json({
    success: true,
    data: newPost,
  })
}
