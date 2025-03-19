import { Camera } from 'expo-camera';

// Camera Types
export enum CameraType {
  front = 'front',
  back = 'back'
}

// Screen Component Types
export type RootStackParamList = {
  Home: undefined;
  VideoUploader: undefined;
  VideoAnalysis: {
    videoId: string;
    taskId: string;
    videoUri?: string;
  };
  Profile: undefined;
  Settings: undefined;
  Auth: undefined;
};

// API Types
export interface TaskResponse {
  task_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
}

// Make the original Camera type from expo-camera available 
export type ExpoCameraType = Camera['props']['type'];

// Define custom component types
export interface VideoData {
  id: string;
  user_id: string;
  title: string;
  description?: string;
  storage_path: string;
  thumbnail_path?: string;
  duration?: number;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
}

// Define function callback types
export type UploadProgressCallback = (progress: number) => void;
export type UploadSuccessCallback = (result: { videoId: string; taskId: string }) => void;
export type UploadErrorCallback = (error: Error) => void;