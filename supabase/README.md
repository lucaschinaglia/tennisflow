# Supabase Setup for TennisFlow

This directory contains the database schema and setup instructions for the TennisFlow application's Supabase backend.

## Getting Started

1. Create a new Supabase project at [supabase.com](https://supabase.com)
2. Navigate to the SQL Editor in your Supabase dashboard
3. Run the `schema.sql` script to set up the database schema

## Database Schema

The TennisFlow application uses the following tables:

- `users` - Extended user profiles (linked to Supabase Auth)
- `videos` - Tennis videos uploaded by users
- `analysis_details` - Detailed analysis results for each frame/timestamp
- `feedback` - System, coach, and user feedback on videos
- `progress` - User progress tracking for different metrics

## Security

The schema includes Row-Level Security (RLS) policies to ensure users can only access their own data or data that has been specifically shared with them.

## Authentication

The application uses Supabase Auth for user authentication. When a new user signs up, a trigger automatically creates a corresponding record in the `users` table.

## Storage Setup

You'll need to configure storage buckets for video storage:

1. In your Supabase dashboard, navigate to Storage
2. Create a new bucket called `videos`
3. Configure the following RLS policies for the bucket:

For the `videos` bucket:

```sql
-- Allow users to view their own videos
CREATE POLICY "Users can view their own videos" 
ON storage.objects FOR SELECT 
USING (bucket_id = 'videos' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Allow users to upload videos to their own folder
CREATE POLICY "Users can upload videos" 
ON storage.objects FOR INSERT 
WITH CHECK (
  bucket_id = 'videos' AND 
  auth.uid()::text = (storage.foldername(name))[1] AND
  (storage.extension(name) = 'mp4' OR 
   storage.extension(name) = 'mov' OR 
   storage.extension(name) = 'avi')
);

-- Allow users to update their own videos
CREATE POLICY "Users can update their own videos" 
ON storage.objects FOR UPDATE 
USING (bucket_id = 'videos' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Allow users to delete their own videos
CREATE POLICY "Users can delete their own videos" 
ON storage.objects FOR DELETE 
USING (bucket_id = 'videos' AND auth.uid()::text = (storage.foldername(name))[1]);
```

## Environment Variables

After setting up your Supabase project, you'll need to add the following environment variables to your app:

```
EXPO_PUBLIC_SUPABASE_URL=your_supabase_url
EXPO_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

You can find these values in the API settings of your Supabase dashboard.