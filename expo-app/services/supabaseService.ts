import 'react-native-url-polyfill/auto';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { createClient } from '@supabase/supabase-js';
import Constants from 'expo-constants';
import { testSupabaseAuth } from '../lib/auth-debugger';

// Get environment variables from Expo config
const getSupabaseUrl = () => {
  console.log('[supabaseService.ts] EXPO_PUBLIC_SUPABASE_URL exists?', !!process.env.EXPO_PUBLIC_SUPABASE_URL);
  if (process.env.EXPO_PUBLIC_SUPABASE_URL) {
    console.log('[supabaseService.ts] Using URL from env var:', process.env.EXPO_PUBLIC_SUPABASE_URL);
    return process.env.EXPO_PUBLIC_SUPABASE_URL;
  }
  // Fallback for development
  console.log('[supabaseService.ts] Using hardcoded URL');
  return 'https://mmjpyrqiemwpoidbmcdg.supabase.co';
};

const getSupabaseAnonKey = () => {
  console.log('[supabaseService.ts] EXPO_PUBLIC_SUPABASE_ANON_KEY exists?', !!process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY);
  if (process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY) {
    console.log('[supabaseService.ts] Using API key from env var');
    return process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
  }
  // Use anon key, not service role key for client-side
  console.log('[supabaseService.ts] Using hardcoded API key');
  return 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1tanB5cnFpZW13cG9pZGJtY2RnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI0MjE3MDAsImV4cCI6MjA1Nzk5NzcwMH0.2eKghe0Qf1RwqDh25CDYGSyDPj9x_YkzotiNxC9gvXs';
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

// DEBUG WARNING: You have two Supabase client initializations - fix this!
console.warn('⚠️ WARNING: Multiple Supabase clients detected. Use lib/supabase.ts for all imports.');
console.warn('⚠️ This can cause authentication issues. Update imports to use a single client instance.');

// Auth functions
export const signIn = async (email: string, password: string) => {
  console.log('Attempting to sign in with email:', email);
  
  // First try with our potentially problematic client
  const result = await supabase.auth.signInWithPassword({ email, password });
  
  if (result.error) {
    console.error('Sign in error:', result.error);
    console.log('Trying with direct test client as fallback...');
    
    // If it fails, try with our guaranteed correct test client
    const directResult = await testSupabaseAuth(email, password);
    
    if (directResult.success) {
      console.log('Direct test succeeded, but regular client failed - this indicates a configuration issue!');
      return { data: directResult.data, error: null };
    }
  } else {
    console.log('Sign in successful. Session exists:', !!result.data.session);
    // Check if session is being set properly
    const { data: sessionData } = await supabase.auth.getSession();
    console.log('Session after login:', sessionData.session ? 'exists' : 'missing');
  }
  
  return result;
};

export const signUp = async (email: string, password: string) => {
  console.log('Attempting to sign up with email:', email);
  const result = await supabase.auth.signUp({ email, password });
  
  if (result.error) {
    console.error('Sign up error:', result.error);
  } else {
    console.log('Sign up successful. Session exists:', !!result.data.session);
  }
  
  return result;
};

export const signOut = async () => {
  console.log('Signing out');
  const result = await supabase.auth.signOut();
  
  if (result.error) {
    console.error('Sign out error:', result.error);
  } else {
    console.log('Sign out successful');
  }
  
  return result;
};

export const resetPassword = async (email: string) => {
  console.log('Sending password reset email to:', email);
  const result = await supabase.auth.resetPasswordForEmail(email);
  
  if (result.error) {
    console.error('Password reset error:', result.error);
  } else {
    console.log('Password reset email sent successfully');
  }
  
  return result;
};

// Check if a user is already logged in
export const getCurrentUser = async () => {
  console.log('Checking for current user');
  const result = await supabase.auth.getUser();
  
  if (result.error) {
    console.error('Get user error:', result.error);
  } else {
    console.log('Current user:', result.data.user ? result.data.user.email : 'No user found');
  }
  
  return result;
};

// Get session data
export const getSession = async () => {
  console.log('Getting current session');
  const result = await supabase.auth.getSession();
  
  if (result.error) {
    console.error('Get session error:', result.error);
  } else {
    console.log('Session exists:', !!result.data.session);
  }
  
  return result;
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