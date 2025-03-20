import { createClient } from '@supabase/supabase-js';
import 'react-native-url-polyfill/auto';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Supabase configuration
// Replace with your own Supabase URL and anon key
const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL || 'https://mmjpyrqiemwpoidbmcdg.supabase.co';
const supabaseAnonKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1tanB5cnFpZW13cG9pZGJtY2RnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIzMzU0NTYsImV4cCI6MjA1NzkxMTQ1Nn0.KzAP_VGleqdf0tH3Mxq6kQlz-s2AoYFGLOA-M4dpAow';

console.log('Supabase URL:', supabaseUrl);
console.log('Supabase Key exists:', !!supabaseAnonKey);

// Create Supabase client with debug logging
export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    storage: AsyncStorage,
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: false,
    debug: __DEV__, // Enable debug logs in development
  },
});

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