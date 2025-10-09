import { NextResponse } from "next/server"
import { mockMessages } from "@/lib/mock-data"

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const conversationId = searchParams.get("conversationId")

  await new Promise((resolve) => setTimeout(resolve, 300))

  const filteredMessages = conversationId
    ? mockMessages.filter((m) => m.conversationId === conversationId)
    : mockMessages

  return NextResponse.json({
    success: true,
    data: filteredMessages,
  })
}

export async function POST(request: Request) {
  const body = await request.json()

  await new Promise((resolve) => setTimeout(resolve, 300))

  const newMessage = {
    id: `msg${Date.now()}`,
    conversationId: body.conversationId,
    senderId: body.senderId,
    content: body.content,
    timestamp: new Date().toISOString(),
    read: false,
    type: "text" as const,
  }

  return NextResponse.json({
    success: true,
    data: newMessage,
  })
}
