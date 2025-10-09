"use client"

import { useEffect } from "react"
import { Sparkles } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

interface PointsToastProps {
  points: number
  reason: string
  show: boolean
  onHide: () => void
}

export function PointsToast({ points, reason, show, onHide }: PointsToastProps) {
  useEffect(() => {
    if (show) {
      const timer = setTimeout(onHide, 3000)
      return () => clearTimeout(timer)
    }
  }, [show, onHide])

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, y: -50, scale: 0.8 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -50, scale: 0.8 }}
          className="fixed top-20 left-1/2 z-50 -translate-x-1/2"
        >
          <div className="flex items-center gap-3 rounded-full border border-accent/30 bg-accent/10 px-6 py-3 backdrop-blur-xl">
            <Sparkles className="h-5 w-5 text-accent" />
            <div>
              <p className="text-sm font-bold text-accent">+{points} points</p>
              <p className="text-xs text-muted-foreground">{reason}</p>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
