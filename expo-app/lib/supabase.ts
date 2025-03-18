import { createClient } from '@supabase/supabase-js';
import 'react-native-url-polyfill/auto';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Supabase configuration
// Replace with your own Supabase URL and anon key
const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY || '';

// Create Supabase client
export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    storage: AsyncStorage,
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: false,
  },
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