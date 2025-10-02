import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
  }
})

// Type exports for your tables
export type User = {
  id: string
  email?: string
  full_name?: string
  username?: string
  avatar_url?: string
  created_at: string
}

export type Pet = {
  id: string
  user_id: string
  name: string
  breed?: string
  age?: number
  size?: string
  temperament?: any
  avatar_url?: string
  created_at: string
}

export type Post = {
  id: string
  user_id: string
  pet_id?: string
  content?: string
  media_urls?: string[]
  location?: any
  visibility?: string
  created_at: string
}

export type Walk = {
  id: string
  user_id: string
  pet_id: string
  distance_km?: number
  duration_minutes?: number
  route_coordinates?: any
  calories_burned?: number
  created_at: string
}