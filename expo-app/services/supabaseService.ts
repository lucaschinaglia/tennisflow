import 'react-native-url-polyfill/auto';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { createClient } from '@supabase/supabase-js';
import Constants from 'expo-constants';

// Get environment variables from Expo config
const getSupabaseUrl = () => {
  if (process.env.EXPO_PUBLIC_SUPABASE_URL) {
    return process.env.EXPO_PUBLIC_SUPABASE_URL;
  }
  // Fallback for development
  return 'https://mmjpyrqiemwpoidbmcdg.supabase.co';
};

const getSupabaseAnonKey = () => {
  if (process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY) {
    return process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
  }
  // Use anon key, not service role key for client-side
  return 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1tanB5cnFpZW13cG9pZGJtY2RnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIzMzU0NTYsImV4cCI6MjA1NzkxMTQ1Nn0.KzAP_VGleqdf0tH3Mxq6kQlz-s2AoYFGLOA-M4dpAow';
};

// Create Supabase client
export const supabase = createClient(getSupabaseUrl(), getSupabaseAnonKey(), {
  auth: {
    storage: AsyncStorage,
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: false,
  },
});

// Auth functions
export const signIn = async (email: string, password: string) => {
  return await supabase.auth.signInWithPassword({ email, password });
};

export const signUp = async (email: string, password: string) => {
  return await supabase.auth.signUp({ email, password });
};

export const signOut = async () => {
  return await supabase.auth.signOut();
};

export const resetPassword = async (email: string) => {
  return await supabase.auth.resetPasswordForEmail(email);
};

// Check if a user is already logged in
export const getCurrentUser = async () => {
  return await supabase.auth.getUser();
};

// Get session data
export const getSession = async () => {
  return await supabase.auth.getSession();
};

// Data functions
export const fetchUserProfile = async (userId: string) => {
  return await supabase
    .from('profiles')
    .select('*')
    .eq('id', userId)
    .single();
};

export const updateUserProfile = async (userId: string, updates: any) => {
  return await supabase
    .from('profiles')
    .update(updates)
    .eq('id', userId);
};

// Storage functions for videos
export const getVideos = async (userId: string) => {
  return await supabase
    .from('videos')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false });
};

export const uploadVideo = async (filePath: string, file: Blob) => {
  return await supabase.storage
    .from('tennis-videos')
    .upload(filePath, file);
};

export const getVideoUrl = async (filePath: string) => {
  return await supabase.storage
    .from('tennis-videos')
    .createSignedUrl(filePath, 3600); // 1 hour expiry
};

export const deleteVideo = async (filePath: string) => {
  return await supabase.storage
    .from('tennis-videos')
    .remove([filePath]);
};

// Check connection
export const checkConnection = async () => {
  try {
    const { data, error } = await supabase.from('videos').select('count', { count: 'exact' }).limit(1);
    if (error) throw error;
    return true;
  } catch (error) {
    console.error('Supabase connection error:', error);
    return false;
  }
};