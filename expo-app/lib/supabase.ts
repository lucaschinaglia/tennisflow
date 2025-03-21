import { createClient } from '@supabase/supabase-js';
import 'react-native-url-polyfill/auto';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Supabase configuration
// Replace with your own Supabase URL and anon key
const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL || 'https://mmjpyrqiemwpoidbmcdg.supabase.co';
const supabaseAnonKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1tanB5cnFpZW13cG9pZGJtY2RnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI0MjE3MDAsImV4cCI6MjA1Nzk5NzcwMH0.2eKghe0Qf1RwqDh25CDYGSyDPj9x_YkzotiNxC9gvXs';

console.log('[lib/supabase.ts] Supabase URL:', supabaseUrl);
console.log('[lib/supabase.ts] Supabase Key:', supabaseAnonKey.substring(0, 10) + '...');
console.log('[lib/supabase.ts] Environment variables loaded:', !!process.env.EXPO_PUBLIC_SUPABASE_URL, !!process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY);

// Create and export a function to get the Supabase client
// This helps with ensuring we use only one instance throughout the app
export function createSupabaseClient() {
  const url = process.env.EXPO_PUBLIC_SUPABASE_URL || 'https://mmjpyrqiemwpoidbmcdg.supabase.co';
  const key = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1tanB5cnFpZW13cG9pZGJtY2RnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI0MjE3MDAsImV4cCI6MjA1Nzk5NzcwMH0.2eKghe0Qf1RwqDh25CDYGSyDPj9x_YkzotiNxC9gvXs';
  
  console.log('[SUPABASE CLIENT] Creating with URL:', url);
  console.log('[SUPABASE CLIENT] Key exists:', !!key);
  
  return createClient(url, key, {
    auth: {
      storage: AsyncStorage,
      autoRefreshToken: true,
      persistSession: true,
      detectSessionInUrl: false,
      debug: __DEV__,
    },
  });
}

// Create a shared instance for import
export const supabase = createSupabaseClient();

// Debug: Log session on client creation
supabase.auth.getSession().then(({ data }) => {
  console.log('Initial session check:', data.session ? 'Session exists' : 'No session');
});

// Define database types for TypeScript
export type Tables = {
  users: {
    Row: {
      id: string;
      created_at: string;
      email: string;
      first_name: string | null;
      last_name: string | null;
      avatar_url: string | null;
    };
    Insert: {
      id?: string;
      created_at?: string;
      email: string;
      first_name?: string | null;
      last_name?: string | null;
      avatar_url?: string | null;
    };
    Update: {
      id?: string;
      created_at?: string;
      email?: string;
      first_name?: string | null;
      last_name?: string | null;
      avatar_url?: string | null;
    };
  };
  videos: {
    Row: {
      id: string;
      created_at: string;
      user_id: string;
      title: string;
      description: string | null;
      video_url: string;
      thumbnail_url: string | null;
      analysis_status: 'pending' | 'processing' | 'completed' | 'failed';
      analysis_results: any | null;
    };
    Insert: {
      id?: string;
      created_at?: string;
      user_id: string;
      title: string;
      description?: string | null;
      video_url: string;
      thumbnail_url?: string | null;
      analysis_status?: 'pending' | 'processing' | 'completed' | 'failed';
      analysis_results?: any | null;
    };
    Update: {
      id?: string;
      created_at?: string;
      user_id?: string;
      title?: string;
      description?: string | null;
      video_url?: string;
      thumbnail_url?: string | null;
      analysis_status?: 'pending' | 'processing' | 'completed' | 'failed';
      analysis_results?: any | null;
    };
  };
};

export type Database = {
  public: {
    Tables: Tables;
  };
};