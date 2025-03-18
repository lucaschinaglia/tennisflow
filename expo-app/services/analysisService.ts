import { supabase } from './supabaseService';
import Constants from 'expo-constants';
import axios from 'axios';
import { Platform } from 'react-native';

// Define API URL based on environment
const getApiUrl = () => {
  // For testing in Expo Go with local dev API
  if (__DEV__) {
    return Platform.OS === 'android' 
      ? 'http://10.0.2.2:8000' // Android simulator uses different address
      : 'http://localhost:8000';
  }
  
  // Production API URL
  return 'https://api.tennisflow.app';
};

const API_URL = getApiUrl();

// Define interfaces for analysis data
export interface PoseKeypoint {
  name: string;
  position: {
    x: number;
    y: number;
  };
  confidence: number;
}

export interface PoseConnection {
  from: string;
  to: string;
}

export interface PoseData {
  keypoints: PoseKeypoint[];
  connections: PoseConnection[];
}

export interface Annotation {
  type: string;
  text: string;
  position: {
    x: number;
    y: number;
  };
  value?: number;
  color: string;
}

export interface FrameData {
  frameNumber: number;
  timestamp: number;
  poseData: PoseData;
  swingPhase: string;
  annotations: Annotation[];
}

export interface SwingMetrics {
  racketSpeed: number;
  hipRotation: number;
  shoulderRotation: number;
  kneeFlexion: number;
  weightTransfer: number;
  balanceScore: number;
  followThrough: number;
  consistency: number;
}

export interface SwingData {
  id: string;
  startTime: number;
  endTime: number;
  swingType: string;
  metrics: SwingMetrics;
  keyFrames: number[];
  score: number;
}

export interface AnalysisSummary {
  swingType: string;
  swingCount: number;
  averageMetrics: SwingMetrics;
  strengths: string[];
  weaknesses: string[];
  improvementSuggestions: string[];
}

export interface AnalysisResult {
  videoId: string;
  duration: number;
  frames: FrameData[];
  summary: AnalysisSummary;
  swings: SwingData[];
}

export interface TaskStatus {
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress?: number;
  error?: string;
  result_url?: string;
}

/**
 * Submit a video for analysis
 * @param videoId The ID of the video in Supabase storage
 * @returns A task ID for checking status
 */
export const submitVideoForAnalysis = async (videoId: string): Promise<string> => {
  try {
    const response = await axios.post(`${API_URL}/analyze`, { video_id: videoId });
    return response.data.task_id;
  } catch (error) {
    console.error('Error submitting video for analysis:', error);
    throw new Error('Failed to submit video for analysis');
  }
};

/**
 * Check the status of an analysis task
 * @param taskId The task ID returned from submitVideoForAnalysis
 * @returns The current status of the task
 */
export const checkAnalysisStatus = async (taskId: string): Promise<TaskStatus> => {
  try {
    const response = await axios.get(`${API_URL}/status/${taskId}`);
    return response.data;
  } catch (error) {
    console.error('Error checking analysis status:', error);
    throw new Error('Failed to check analysis status');
  }
};

/**
 * Get analysis results for a video
 * @param videoId The ID of the analyzed video
 * @returns The complete analysis results
 */
export const getVideoAnalysis = async (videoId: string): Promise<AnalysisResult> => {
  try {
    const response = await axios.get(`${API_URL}/video/${videoId}/analysis`);
    return response.data;
  } catch (error) {
    console.error('Error fetching video analysis:', error);
    throw new Error('Failed to fetch video analysis');
  }
};

/**
 * Upload a video to Supabase storage and submit it for analysis
 * @param uri Local URI of the video file
 * @param fileName Desired filename in storage
 * @returns Object with videoId and taskId
 */
export const uploadAndAnalyzeVideo = async (
  uri: string, 
  fileName: string
): Promise<{ videoId: string, taskId: string }> => {
  try {
    // First upload to Supabase
    const response = await fetch(uri);
    const blob = await response.blob();
    
    const fileExt = fileName.split('.').pop();
    const filePath = `${Date.now()}.${fileExt}`;
    
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from('tennis-videos')
      .upload(filePath, blob);
    
    if (uploadError) {
      throw new Error(`Upload error: ${uploadError.message}`);
    }
    
    const videoId = uploadData.path;
    
    // Then submit to processing API
    const uploadResponse = await axios.post(`${API_URL}/upload`, {
      video_path: videoId,
      filename: fileName
    });
    
    return {
      videoId,
      taskId: uploadResponse.data.task_id
    };
  } catch (error) {
    console.error('Error in upload and analyze:', error);
    throw new Error('Failed to upload and analyze video');
  }
};

/**
 * Check the health status of the API
 * @returns Health status object
 */
export const checkApiHealth = async (): Promise<{
  status: string;
  redis: boolean;
  supabase: boolean;
}> => {
  try {
    const response = await axios.get(`${API_URL}/health`);
    return response.data;
  } catch (error) {
    console.error('API health check failed:', error);
    return {
      status: 'error',
      redis: false, 
      supabase: false
    };
  }
};