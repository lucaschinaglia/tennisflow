-- Create schema for TennisFlow application

-- Users table extension
-- Builds on the auth.users table provided by Supabase Auth
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY REFERENCES auth.users ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  email TEXT NOT NULL,
  first_name TEXT,
  last_name TEXT,
  avatar_url TEXT
);

-- Videos table for storing user-uploaded tennis videos
CREATE TABLE IF NOT EXISTS videos (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  description TEXT,
  video_url TEXT NOT NULL,
  thumbnail_url TEXT,
  analysis_status TEXT CHECK (analysis_status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'pending',
  analysis_results JSONB
);

-- Analysis details table for storing specific swing analysis data
CREATE TABLE IF NOT EXISTS analysis_details (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  frame_number INTEGER,
  timestamp FLOAT,
  pose_data JSONB,
  swing_phase TEXT,
  metrics JSONB,
  annotations JSONB
);

-- User feedback table for storing system feedback and user comments
CREATE TABLE IF NOT EXISTS feedback (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  feedback_type TEXT CHECK (feedback_type IN ('system', 'coach', 'user')),
  content TEXT NOT NULL,
  timestamp FLOAT,
  is_public BOOLEAN DEFAULT FALSE
);

-- Progress tracking table for monitoring user improvement
CREATE TABLE IF NOT EXISTS progress (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  metric_name TEXT NOT NULL,
  metric_value FLOAT NOT NULL,
  video_id UUID REFERENCES videos(id) ON DELETE SET NULL
);

-- Configure Row-Level Security (RLS)
-- Users can only access their own data

-- Users table RLS
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE POLICY users_policy ON users
  USING (id = auth.uid());

-- Videos table RLS
ALTER TABLE videos ENABLE ROW LEVEL SECURITY;

CREATE POLICY videos_select_policy ON videos
  FOR SELECT USING (user_id = auth.uid());

CREATE POLICY videos_insert_policy ON videos
  FOR INSERT WITH CHECK (user_id = auth.uid());

CREATE POLICY videos_update_policy ON videos
  FOR UPDATE USING (user_id = auth.uid());

CREATE POLICY videos_delete_policy ON videos
  FOR DELETE USING (user_id = auth.uid());

-- Analysis details table RLS
ALTER TABLE analysis_details ENABLE ROW LEVEL SECURITY;

CREATE POLICY analysis_details_policy ON analysis_details
  USING (video_id IN (SELECT id FROM videos WHERE user_id = auth.uid()));

-- Feedback table RLS
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

CREATE POLICY feedback_select_policy ON feedback
  FOR SELECT USING (
    user_id = auth.uid() OR 
    video_id IN (SELECT id FROM videos WHERE user_id = auth.uid()) OR
    is_public = TRUE
  );

CREATE POLICY feedback_insert_policy ON feedback
  FOR INSERT WITH CHECK (user_id = auth.uid());

CREATE POLICY feedback_update_policy ON feedback
  FOR UPDATE USING (user_id = auth.uid());

CREATE POLICY feedback_delete_policy ON feedback
  FOR DELETE USING (user_id = auth.uid());

-- Progress table RLS
ALTER TABLE progress ENABLE ROW LEVEL SECURITY;

CREATE POLICY progress_policy ON progress
  USING (user_id = auth.uid());

-- Create function to handle new user signup
CREATE OR REPLACE FUNCTION public.handle_new_user() 
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.users (id, email, avatar_url)
  VALUES (NEW.id, NEW.email, NEW.raw_user_meta_data->>'avatar_url');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger the function every time a user signs up
CREATE OR REPLACE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();