import { NextResponse } from "next/server"
import { mockHealthRecords } from "@/lib/mock-data"

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const petId = searchParams.get("petId")

  await new Promise((resolve) => setTimeout(resolve, 300))

  const filteredRecords = petId ? mockHealthRecords.filter((r) => r.petId === petId) : mockHealthRecords

  return NextResponse.json({
    success: true,
    data: filteredRecords,
  })
}

export async function POST(request: Request) {
  const body = await request.json()

  await new Promise((resolve) => setTimeout(resolve, 400))

  const newRecord = {
    id: `h${Date.now()}`,
    ...body,
    date: new Date(body.date).toISOString(),
  }

  return NextResponse.json({
    success: true,
    data: newRecord,
  })
}
