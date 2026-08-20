"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { X, Camera, Video, RotateCw, Check } from "lucide-react"
import { useRouter } from "next/navigation"
import { cn } from "@/lib/utils"

type CaptureMode = "photo" | "video"
type CameraState = "preview" | "captured"

export default function CameraPage() {
  const router = useRouter()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const [stream, setStream] = useState<MediaStream | null>(null)
  const [mode, setMode] = useState<CaptureMode>("photo")
  const [state, setState] = useState<CameraState>("preview")
  const [capturedMedia, setCapturedMedia] = useState<string | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [facingMode, setFacingMode] = useState<"user" | "environment">("environment")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    startCamera()
    return () => {
      stopCamera()
    }
  }, [facingMode])

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode },
        audio: mode === "video",
      })
      setStream(mediaStream)
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
      setError(null)
    } catch (err) {
      console.error("Error accessing camera:", err)
      setError("Unable to access camera. Please check permissions.")
    }
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
  }

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const ctx = canvas.getContext("2d")
    if (ctx) {
      ctx.drawImage(video, 0, 0)
      const imageUrl = canvas.toDataURL("image/jpeg")
      setCapturedMedia(imageUrl)
      setState("captured")
    }
  }

  const startRecording = () => {
    if (!stream) return

    chunksRef.current = []
    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: "video/webm",
    })

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        chunksRef.current.push(e.data)
      }
    }

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" })
      const videoUrl = URL.createObjectURL(blob)
      setCapturedMedia(videoUrl)
      setState("captured")
    }

    mediaRecorderRef.current = mediaRecorder
    mediaRecorder.start()
    setIsRecording(true)
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const handleCapture = () => {
    if (mode === "photo") {
      capturePhoto()
    } else {
      if (isRecording) {
        stopRecording()
      } else {
        startRecording()
      }
    }
  }

  const handleRetake = () => {
    setCapturedMedia(null)
    setState("preview")
    setIsRecording(false)
  }

  const handlePost = () => {
    // In a real app, this would upload the media and create a post
    router.push("/")
  }

  const toggleCamera = () => {
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"))
  }

  if (error) {
    return (
      <div className="fixed inset-0 bg-background flex items-center justify-center p-4">
        <div className="text-center space-y-4">
          <Camera className="w-16 h-16 mx-auto text-muted-foreground" />
          <p className="text-muted-foreground">{error}</p>
          <Button onClick={() => router.back()}>Go Back</Button>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 bg-background z-50">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-4">
        <Button variant="ghost" size="icon" onClick={() => router.back()} className="bg-background/50 backdrop-blur">
          <X className="w-6 h-6" />
        </Button>
        {state === "preview" && (
          <Button variant="ghost" size="icon" onClick={toggleCamera} className="bg-background/50 backdrop-blur">
            <RotateCw className="w-6 h-6" />
          </Button>
        )}
      </div>

      {/* Camera Preview or Captured Media */}
      <div className="relative w-full h-full">
        {state === "preview" ? (
          <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-black">
            {mode === "photo" ? (
              <img src={capturedMedia || ""} alt="Captured" className="max-w-full max-h-full object-contain" />
            ) : (
              <video src={capturedMedia || ""} controls className="max-w-full max-h-full object-contain" />
            )}
          </div>
        )}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {/* Bottom Controls */}
      <div className="absolute bottom-0 left-0 right-0 z-10 pb-8">
        {state === "preview" ? (
          <div className="flex items-center justify-center gap-8 px-4">
            {/* Mode Toggle */}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setMode("photo")}
              className={cn("bg-background/50 backdrop-blur", mode === "photo" && "text-primary")}
            >
              <Camera className="w-6 h-6" />
            </Button>

            {/* Capture Button */}
            <button
              onClick={handleCapture}
              className={cn(
                "w-20 h-20 rounded-full border-4 border-white flex items-center justify-center transition-all",
                isRecording ? "bg-red-500 scale-90" : "bg-white/30 backdrop-blur hover:scale-105",
              )}
            >
              {isRecording && <div className="w-8 h-8 bg-white rounded-sm" />}
            </button>

            {/* Video Mode Toggle */}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setMode("video")}
              className={cn("bg-background/50 backdrop-blur", mode === "video" && "text-primary")}
            >
              <Video className="w-6 h-6" />
            </Button>
          </div>
        ) : (
          <div className="flex items-center justify-center gap-8 px-4">
            <Button variant="ghost" size="lg" onClick={handleRetake} className="bg-background/50 backdrop-blur gap-2">
              <X className="w-5 h-5" />
              Retake
            </Button>
            <Button size="lg" onClick={handlePost} className="gap-2">
              <Check className="w-5 h-5" />
              Post
            </Button>
          </div>
        )}
      </div>

      {/* Recording Indicator */}
      {isRecording && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 px-4 py-2 rounded-full bg-red-500 text-white">
          <div className="w-3 h-3 bg-white rounded-full animate-pulse" />
          <span className="text-sm font-medium">Recording</span>
        </div>
      )}
    </div>
  )
}
