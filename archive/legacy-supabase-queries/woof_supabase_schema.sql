CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- Users table: Basic profile with role
CREATE TABLE public.users (
  id UUID PRIMARY KEY REFERENCES auth.users (id) ON DELETE CASCADE,
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user','moderator','admin')),
  xp INTEGER NOT NULL DEFAULT 0,
  level INTEGER NOT NULL DEFAULT 1,
  premium_tier TEXT,
  premium_expiration TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now()
);
COMMENT ON TABLE public.users IS 'App users (roles: user, moderator, admin).';

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view profiles" ON public.users
  FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "Users can modify own profile" ON public.users
  FOR UPDATE USING (auth.uid() = id OR exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator')))
  WITH CHECK (auth.uid() = id OR exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator')));
CREATE POLICY "Users can insert self" ON public.users
  FOR INSERT WITH CHECK (auth.uid() = id);

-- Pets table
CREATE TABLE public.pets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  species TEXT NOT NULL CHECK (species IN ('dog','cat','other')),
  breed TEXT,
  birth_date DATE,
  size_category TEXT CHECK (size_category IN ('small','medium','large')),
  happiness_score INTEGER DEFAULT 50,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON public.pets(owner_id);
COMMENT ON TABLE public.pets IS 'Pet profiles (owned by users).';

ALTER TABLE public.pets ENABLE ROW LEVEL SECURITY;
CREATE POLICY "All users can read pets" ON public.pets
  FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "Pet owners can insert" ON public.pets
  FOR INSERT WITH CHECK (owner_id = auth.uid());
CREATE POLICY "Pet owners can update" ON public.pets
  FOR UPDATE USING (owner_id = auth.uid() OR exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator')))
  WITH CHECK (owner_id = auth.uid() OR exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator')));
CREATE POLICY "Pet owners can delete" ON public.pets
  FOR DELETE USING (owner_id = auth.uid() OR exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator')));

-- Walks
CREATE TABLE public.walks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  pet_id UUID NOT NULL REFERENCES public.pets(id) ON DELETE CASCADE,
  walked_at TIMESTAMPTZ DEFAULT now(),
  duration_minutes INT,
  distance_km NUMERIC(6,2),
  route GEOGRAPHY,
  route_preview TEXT
);
CREATE INDEX ON public.walks(user_id);
CREATE INDEX ON public.walks(pet_id);
COMMENT ON TABLE public.walks IS 'Recorded walks with GPS data for pets.';

ALTER TABLE public.walks ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Only owner can access walk" ON public.walks
  FOR SELECT USING (user_id = auth.uid());
CREATE POLICY "Only owner can insert walk" ON public.walks
  FOR INSERT WITH CHECK (
    user_id = auth.uid()
    AND (SELECT owner_id FROM public.pets WHERE public.pets.id = pet_id) = auth.uid()
  );
CREATE POLICY "Only owner can update walk" ON public.walks
  FOR UPDATE USING (user_id = auth.uid()) WITH CHECK (user_id = auth.uid());
CREATE POLICY "Only owner can delete walk" ON public.walks
  FOR DELETE USING (user_id = auth.uid());

-- Meetups
CREATE TABLE public.meetups (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  host_user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  description TEXT,
  event_time TIMESTAMPTZ NOT NULL,
  location_name TEXT,
  location_lat DOUBLE PRECISION,
  location_lon DOUBLE PRECISION,
  pet_type_filter TEXT CHECK (pet_type_filter IN ('dog','cat','any')) DEFAULT 'any',
  size_filter TEXT CHECK (size_filter IN ('small','medium','large','any')) DEFAULT 'any',
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON public.meetups(event_time);
COMMENT ON TABLE public.meetups IS 'Pet meetups/events (organizer = host_user_id, with pet filters).';

ALTER TABLE public.meetups ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view meetups" ON public.meetups
  FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "Meetup host can update" ON public.meetups
  FOR UPDATE USING (host_user_id = auth.uid() OR exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator')))
  WITH CHECK (host_user_id = auth.uid() OR exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator')));
CREATE POLICY "Meetup host can delete" ON public.meetups
  FOR DELETE USING (host_user_id = auth.uid() OR exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator')));
CREATE POLICY "Users can create meetup" ON public.meetups
  FOR INSERT WITH CHECK (host_user_id = auth.uid());

-- Meetup attendees
CREATE TABLE public.meetup_attendees (
  meetup_id UUID REFERENCES public.meetups(id) ON DELETE CASCADE,
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
  pet_id UUID REFERENCES public.pets(id),
  joined_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY(meetup_id, user_id)
);
CREATE INDEX ON public.meetup_attendees(meetup_id);
CREATE INDEX ON public.meetup_attendees(user_id);
COMMENT ON TABLE public.meetup_attendees IS 'RSVP records linking users (and their pet) to meetups.';

ALTER TABLE public.meetup_attendees ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view attendees" ON public.meetup_attendees
  FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "User can RSVP to meetup" ON public.meetup_attendees
  FOR INSERT WITH CHECK (
    user_id = auth.uid()
    AND (SELECT owner_id FROM public.pets WHERE public.pets.id = pet_id) = auth.uid()
  );
CREATE POLICY "User can cancel RSVP" ON public.meetup_attendees
  FOR DELETE USING (
    user_id = auth.uid()
    OR exists(SELECT 1 FROM public.meetups m WHERE m.id = meetup_id AND m.host_user_id = auth.uid())
    OR exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator'))
  );

-- Badges
CREATE TABLE public.badges (
  id SERIAL PRIMARY KEY,
  code TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  icon_url TEXT,
  criteria TEXT
);
COMMENT ON TABLE public.badges IS 'Master list of all badge achievements users can earn.';

ALTER TABLE public.badges ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can read badges" ON public.badges
  FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "Admin can insert badges" ON public.badges
  FOR INSERT WITH CHECK ( exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role = 'admin') );
CREATE POLICY "Admin can update badges" ON public.badges
  FOR UPDATE USING ( exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role = 'admin') )
  WITH CHECK ( exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role = 'admin') );

-- User badges
CREATE TABLE public.user_badges (
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
  badge_id INT REFERENCES public.badges(id) ON DELETE CASCADE,
  earned_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY(user_id, badge_id)
);
CREATE INDEX ON public.user_badges(badge_id);
COMMENT ON TABLE public.user_badges IS 'Junction of users to badges they have earned.';

ALTER TABLE public.user_badges ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view user_badges" ON public.user_badges
  FOR SELECT USING (auth.role() = 'authenticated');

-- Posts
CREATE TABLE public.posts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  author_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  content TEXT,
  image_urls TEXT[] DEFAULT ARRAY[]::TEXT[],
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON public.posts(author_id);
COMMENT ON TABLE public.posts IS 'Social feed posts by users (text and media).';

ALTER TABLE public.posts ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can read posts" ON public.posts
  FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "Users can create post" ON public.posts
  FOR INSERT WITH CHECK (author_id = auth.uid());
CREATE POLICY "Edit own posts" ON public.posts
  FOR UPDATE USING (author_id = auth.uid())
  WITH CHECK (author_id = auth.uid());
CREATE POLICY "Delete own or moderated posts" ON public.posts
  FOR DELETE USING (
    author_id = auth.uid()
    OR exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator'))
  );

-- Comments
CREATE TABLE public.comments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  post_id UUID NOT NULL REFERENCES public.posts(id) ON DELETE CASCADE,
  author_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON public.comments(post_id);
CREATE INDEX ON public.comments(author_id);
COMMENT ON TABLE public.comments IS 'Comments on feed posts.';

ALTER TABLE public.comments ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can read comments" ON public.comments
  FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "Users can add comment" ON public.comments
  FOR INSERT WITH CHECK (author_id = auth.uid());
CREATE POLICY "Delete own or mod comments" ON public.comments
  FOR DELETE USING (
    author_id = auth.uid()
    OR exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator'))
  );

-- Likes
CREATE TABLE public.likes (
  post_id UUID NOT NULL REFERENCES public.posts(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  liked_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY(post_id, user_id)
);
CREATE INDEX ON public.likes(user_id);
COMMENT ON TABLE public.likes IS 'Tracks which users liked which posts.';

ALTER TABLE public.likes ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view likes" ON public.likes
  FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "User can like post" ON public.likes
  FOR INSERT WITH CHECK (user_id = auth.uid());
CREATE POLICY "User can unlike post" ON public.likes
  FOR DELETE USING (user_id = auth.uid());

-- Meetup chat messages
CREATE TABLE public.chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  meetup_id UUID NOT NULL REFERENCES public.meetups(id) ON DELETE CASCADE,
  author_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  message TEXT NOT NULL,
  sent_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON public.chat_messages(meetup_id);
CREATE INDEX ON public.chat_messages(author_id);
COMMENT ON TABLE public.chat_messages IS 'Group chat messages for meetups/events.';

ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Meetup members can read messages" ON public.chat_messages
  FOR SELECT USING (
    exists(SELECT 1 FROM public.meetup_attendees ma 
           WHERE ma.meetup_id = meetup_id AND ma.user_id = auth.uid())
    OR exists(SELECT 1 FROM public.meetups m 
           WHERE m.id = meetup_id AND m.host_user_id = auth.uid())
  );
CREATE POLICY "Meetup members can send messages" ON public.chat_messages
  FOR INSERT WITH CHECK (
    author_id = auth.uid() 
    AND exists(SELECT 1 FROM public.meetup_attendees ma 
               WHERE ma.meetup_id = meetup_id AND ma.user_id = auth.uid())
  );
CREATE POLICY "Delete chat message" ON public.chat_messages
  FOR DELETE USING (
    author_id = auth.uid() 
    OR exists(SELECT 1 FROM public.meetups m WHERE m.id = meetup_id AND m.host_user_id = auth.uid())
    OR exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator'))
  );

-- Premium tiers
CREATE TABLE public.premium_tiers (
  code TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  monthly_price_cents INT,
  benefits TEXT
);
COMMENT ON TABLE public.premium_tiers IS 'Available premium membership tiers.';

ALTER TABLE public.premium_tiers ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can read tiers" ON public.premium_tiers
  FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "Admin can modify tiers" ON public.premium_tiers
  FOR INSERT WITH CHECK ( exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role = 'admin') );
CREATE POLICY "Admin can update tiers" ON public.premium_tiers
  FOR UPDATE USING ( exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role = 'admin') )
  WITH CHECK ( exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role = 'admin') );

-- User subscriptions
CREATE TABLE public.user_subscriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  tier_code TEXT NOT NULL REFERENCES public.premium_tiers(code),
  status TEXT NOT NULL CHECK (status IN ('active','canceled','expired')), 
  start_date TIMESTAMPTZ DEFAULT now(),
  end_date TIMESTAMPTZ
);
CREATE INDEX ON public.user_subscriptions(user_id);
COMMENT ON TABLE public.user_subscriptions IS 'Active/past premium subscriptions for users (one per subscription cycle).';

ALTER TABLE public.user_subscriptions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "View own subscription" ON public.user_subscriptions
  FOR SELECT USING (user_id = auth.uid());

-- Activity suggestions
CREATE TABLE public.activity_suggestions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  category TEXT NOT NULL CHECK (category IN ('dog_park','cat_cafe','trail','pet_friendly_cafe','vet','other')),
  description TEXT,
  address TEXT,
  latitude DOUBLE PRECISION,
  longitude DOUBLE PRECISION,
  added_by UUID REFERENCES public.users(id),
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON public.activity_suggestions(category);
COMMENT ON TABLE public.activity_suggestions IS 'Catalog of pet-friendly activities/venues (for suggestions).';

ALTER TABLE public.activity_suggestions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view suggestions" ON public.activity_suggestions
  FOR SELECT USING (auth.role() = 'authenticated');
CREATE POLICY "Admin can add suggestions" ON public.activity_suggestions
  FOR INSERT WITH CHECK ( exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator')) );
CREATE POLICY "Admin can update suggestions" ON public.activity_suggestions
  FOR UPDATE USING ( exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator')) )
  WITH CHECK ( exists(SELECT 1 FROM public.users u WHERE u.id = auth.uid() AND u.role IN ('admin','moderator')) );

-- AI Chatbot sessions and messages
CREATE TABLE public.ai_chat_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  pet_id UUID REFERENCES public.pets(id),
  started_at TIMESTAMPTZ DEFAULT now(),
  feedback_rating INT,
  feedback_comment TEXT
);
CREATE INDEX ON public.ai_chat_sessions(user_id);
COMMENT ON TABLE public.ai_chat_sessions IS 'AI Assistant chat sessions per user (optionally tied to a pet).';

CREATE TABLE public.ai_chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES public.ai_chat_sessions(id) ON DELETE CASCADE,
  sender TEXT CHECK (sender IN ('user','assistant')) NOT NULL,
  message TEXT NOT NULL,
  sent_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON public.ai_chat_messages(session_id);
COMMENT ON TABLE public.ai_chat_messages IS 'Messages exchanged in AI chat sessions (user queries and assistant responses).';

ALTER TABLE public.ai_chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_chat_messages ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can access own AI sessions" ON public.ai_chat_sessions
  FOR SELECT USING (user_id = auth.uid());
CREATE POLICY "Users can start AI session" ON public.ai_chat_sessions
  FOR INSERT WITH CHECK (user_id = auth.uid());
CREATE POLICY "Users can give feedback on session" ON public.ai_chat_sessions
  FOR UPDATE USING (user_id = auth.uid()) WITH CHECK (user_id = auth.uid());
CREATE POLICY "Users access own AI messages" ON public.ai_chat_messages
  FOR SELECT USING (
    exists(SELECT 1 FROM public.ai_chat_sessions s WHERE s.id = session_id AND s.user_id = auth.uid())
  );
CREATE POLICY "Users send AI messages" ON public.ai_chat_messages
  FOR INSERT WITH CHECK (
    sender = 'user' 
    AND exists(SELECT 1 FROM public.ai_chat_sessions s WHERE s.id = session_id AND s.user_id = auth.uid())
  );