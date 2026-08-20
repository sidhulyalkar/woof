import { NextResponse } from "next/server"
import { mockEvents } from "@/lib/mock-data"

export async function GET() {
  await new Promise((resolve) => setTimeout(resolve, 400))

  return NextResponse.json({
    success: true,
    data: mockEvents,
  })
}

export async function POST(request: Request) {
  const body = await request.json()

  await new Promise((resolve) => setTimeout(resolve, 500))

  const newEvent = {
    id: `e${Date.now()}`,
    ...body,
    attendees: [],
    datetime: new Date(body.datetime).toISOString(),
  }

  return NextResponse.json({
    success: true,
    data: newEvent,
  })
}
