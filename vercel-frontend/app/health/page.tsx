"use client"

import { useState } from "react"
import { Heart, Plus, Pill, Syringe, Weight, TrendingUp } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { BottomNav } from "@/components/bottom-nav"
import type { HealthRecord } from "@/lib/types"

export default function HealthPage() {
  const [selectedPet, setSelectedPet] = useState("p1")

  // Mock health records
  const healthRecords: HealthRecord[] = [
    {
      id: "h1",
      petId: "p1",
      type: "vet-visit",
      date: "2024-03-10",
      title: "Annual Checkup",
      description: "Routine examination, all vitals normal",
      veterinarian: "Dr. Sarah Johnson",
      nextDue: "2025-03-10",
    },
    {
      id: "h2",
      petId: "p1",
      type: "vaccination",
      date: "2024-03-10",
      title: "Rabies Vaccine",
      veterinarian: "Dr. Sarah Johnson",
      nextDue: "2027-03-10",
    },
    {
      id: "h3",
      petId: "p1",
      type: "medication",
      date: "2024-03-01",
      title: "Flea & Tick Prevention",
      metadata: {
        medication: "NexGard",
        dosage: "1 chewable",
        frequency: "Monthly",
      },
      nextDue: "2024-04-01",
    },
    {
      id: "h4",
      petId: "p1",
      type: "weight",
      date: "2024-03-10",
      title: "Weight Check",
      metadata: {
        weight: 45.2,
      },
    },
  ]

  const weightHistory = [
    { date: "2024-01-10", weight: 44.5 },
    { date: "2024-02-10", weight: 44.8 },
    { date: "2024-03-10", weight: 45.2 },
  ]

  const getRecordIcon = (type: HealthRecord["type"]) => {
    switch (type) {
      case "vet-visit":
        return <Heart className="h-5 w-5 text-red-400" />
      case "vaccination":
        return <Syringe className="h-5 w-5 text-blue-400" />
      case "medication":
        return <Pill className="h-5 w-5 text-purple-400" />
      case "weight":
        return <Weight className="h-5 w-5 text-green-400" />
    }
  }

  const upcomingRecords = healthRecords.filter((r) => r.nextDue && new Date(r.nextDue) > new Date())

  return (
    <div className="min-h-screen bg-background pb-20">
      {/* Header */}
      <div className="sticky top-0 z-10 border-b border-border/40 bg-background/80 backdrop-blur-xl">
        <div className="flex items-center justify-between px-4 py-4">
          <div className="flex items-center gap-3">
            <Heart className="h-6 w-6 text-accent" />
            <h1 className="text-xl font-bold">Health</h1>
          </div>
          <Button size="sm" className="gap-2">
            <Plus className="h-4 w-4" />
            Add Record
          </Button>
        </div>

        {/* Pet Selector */}
        <div className="px-4 pb-3">
          <div className="flex items-center gap-3 rounded-xl border border-border/40 bg-card/50 p-3">
            <Avatar className="h-10 w-10 border-2 border-border">
              <AvatarImage src="/border-collie.jpg" alt="Charlie" />
              <AvatarFallback>C</AvatarFallback>
            </Avatar>
            <div className="flex-1">
              <p className="font-semibold">Charlie</p>
              <p className="text-xs text-muted-foreground">Border Collie • 3 years</p>
            </div>
          </div>
        </div>
      </div>

      <div className="p-4 space-y-6">
        {/* Upcoming */}
        {upcomingRecords.length > 0 && (
          <div className="space-y-3">
            <h2 className="text-lg font-semibold">Upcoming</h2>
            {upcomingRecords.map((record) => (
              <Card key={record.id} className="glass p-4">
                <div className="flex items-start gap-3">
                  {getRecordIcon(record.type)}
                  <div className="flex-1 min-w-0">
                    <p className="font-semibold">{record.title}</p>
                    <p className="text-sm text-muted-foreground">
                      Due: {new Date(record.nextDue!).toLocaleDateString()}
                    </p>
                  </div>
                  <Badge variant="outline" className="text-xs">
                    {Math.ceil((new Date(record.nextDue!).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24))}{" "}
                    days
                  </Badge>
                </div>
              </Card>
            ))}
          </div>
        )}

        {/* Weight Tracking */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Weight Tracking</h2>
            <Button variant="ghost" size="sm" className="gap-2">
              <TrendingUp className="h-4 w-4" />
              View Chart
            </Button>
          </div>
          <Card className="glass p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-sm text-muted-foreground">Current Weight</p>
                <p className="text-3xl font-bold">{weightHistory[weightHistory.length - 1].weight} lbs</p>
              </div>
              <Weight className="h-12 w-12 text-accent" />
            </div>
            <div className="space-y-2">
              {weightHistory.map((entry, index) => (
                <div key={entry.date} className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">{new Date(entry.date).toLocaleDateString()}</span>
                  <span className="font-medium">{entry.weight} lbs</span>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Records */}
        <Tabs defaultValue="all" className="w-full">
          <TabsList className="w-full justify-start">
            <TabsTrigger value="all">All</TabsTrigger>
            <TabsTrigger value="vet-visit">Vet Visits</TabsTrigger>
            <TabsTrigger value="vaccination">Vaccines</TabsTrigger>
            <TabsTrigger value="medication">Medications</TabsTrigger>
          </TabsList>

          <TabsContent value="all" className="mt-4 space-y-2">
            {healthRecords.map((record) => (
              <Card key={record.id} className="glass p-4">
                <div className="flex items-start gap-3">
                  {getRecordIcon(record.type)}
                  <div className="flex-1 min-w-0">
                    <p className="font-semibold">{record.title}</p>
                    <p className="text-sm text-muted-foreground">{new Date(record.date).toLocaleDateString()}</p>
                    {record.description && <p className="mt-1 text-sm text-muted-foreground">{record.description}</p>}
                    {record.veterinarian && (
                      <p className="mt-1 text-xs text-muted-foreground">Dr. {record.veterinarian}</p>
                    )}
                    {record.metadata && (
                      <div className="mt-2 space-y-1">
                        {record.metadata.medication && (
                          <p className="text-sm">
                            <span className="text-muted-foreground">Medication:</span> {record.metadata.medication}
                          </p>
                        )}
                        {record.metadata.dosage && (
                          <p className="text-sm">
                            <span className="text-muted-foreground">Dosage:</span> {record.metadata.dosage}
                          </p>
                        )}
                        {record.metadata.frequency && (
                          <p className="text-sm">
                            <span className="text-muted-foreground">Frequency:</span> {record.metadata.frequency}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </Card>
            ))}
          </TabsContent>

          <TabsContent value="vet-visit" className="mt-4 space-y-2">
            {healthRecords
              .filter((r) => r.type === "vet-visit")
              .map((record) => (
                <Card key={record.id} className="glass p-4">
                  <div className="flex items-start gap-3">
                    {getRecordIcon(record.type)}
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold">{record.title}</p>
                      <p className="text-sm text-muted-foreground">{new Date(record.date).toLocaleDateString()}</p>
                      {record.description && <p className="mt-1 text-sm text-muted-foreground">{record.description}</p>}
                    </div>
                  </div>
                </Card>
              ))}
          </TabsContent>

          <TabsContent value="vaccination" className="mt-4 space-y-2">
            {healthRecords
              .filter((r) => r.type === "vaccination")
              .map((record) => (
                <Card key={record.id} className="glass p-4">
                  <div className="flex items-start gap-3">
                    {getRecordIcon(record.type)}
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold">{record.title}</p>
                      <p className="text-sm text-muted-foreground">{new Date(record.date).toLocaleDateString()}</p>
                      {record.nextDue && (
                        <p className="mt-1 text-sm text-accent">
                          Next due: {new Date(record.nextDue).toLocaleDateString()}
                        </p>
                      )}
                    </div>
                  </div>
                </Card>
              ))}
          </TabsContent>

          <TabsContent value="medication" className="mt-4 space-y-2">
            {healthRecords
              .filter((r) => r.type === "medication")
              .map((record) => (
                <Card key={record.id} className="glass p-4">
                  <div className="flex items-start gap-3">
                    {getRecordIcon(record.type)}
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold">{record.title}</p>
                      <p className="text-sm text-muted-foreground">{new Date(record.date).toLocaleDateString()}</p>
                      {record.metadata && (
                        <div className="mt-2 space-y-1">
                          {record.metadata.medication && (
                            <p className="text-sm">
                              <span className="text-muted-foreground">Medication:</span> {record.metadata.medication}
                            </p>
                          )}
                          {record.metadata.frequency && (
                            <p className="text-sm">
                              <span className="text-muted-foreground">Frequency:</span> {record.metadata.frequency}
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </Card>
              ))}
          </TabsContent>
        </Tabs>
      </div>

      <BottomNav />
    </div>
  )
}
