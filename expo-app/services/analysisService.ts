import { supabase } from './supabaseService';
import Constants from 'expo-constants';
import axios from 'axios';
import { Platform } from 'react-native';

// Define API URL based on environment
const getApiUrl = () => {
  // For testing in Expo Go with local dev API
  if (__DEV__) {
    // Using direct IP address works better than localhost in many cases
    const devApiUrl = Platform.OS === 'android' 
      ? 'http://10.0.2.2:8001' // Android simulator uses this special address
      : 'http://192.168.0.6:8001'; // Use computer's actual local IP address on your network
      
    console.log("Using development API URL:", devApiUrl);
    return devApiUrl;
  }
  
  // Check for environment variable for non-dev environments
  const envApiUrl = Constants.expoConfig?.extra?.apiUrl || process.env.EXPO_PUBLIC_ANALYSIS_API_URL;
  if (envApiUrl) {
    console.log("Using API URL from environment:", envApiUrl);
    return envApiUrl;
  }
  
  // Production API URL
  return 'https://api.tennisflow.app';
};

const API_URL = getApiUrl();
console.log("Using API URL:", API_URL);

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
    console.log(`Checking status for task: ${taskId}`);
    
    // First check if API is healthy
    const healthStatus = await checkApiHealth();
    if (healthStatus.status !== 'healthy' && healthStatus.status !== 'ok') {
      console.warn('API health check failed before status check');
      // Return a synthetic queued status when API is unhealthy
      return {
        status: 'queued',
        progress: 0,
        error: 'Analysis service temporarily unavailable, will retry'
      };
    }
    
    const response = await axios.get(`${API_URL}/status/${taskId}`, {
      // Add a longer timeout for status check
      timeout: 10000,
      // Add custom error handling for the response
      validateStatus: (status) => {
        // Consider only network failures as errors (not server errors)
        return status >= 200 && status < 300 || status >= 400 && status < 600;
      }
    });
    
    // If we got a server error but with a valid response body, try to use it
    if (response.status >= 400) {
      console.warn(`Status endpoint returned error code: ${response.status}`);
      
      // If we have some data, try to use it
      if (response.data) {
        console.log('Response data despite error:', response.data);
        
        // If it has a status field, use it
        if (response.data.status) {
          return response.data;
        }
      }
      
      // Default to queued status when server is having issues
      return {
        status: 'queued',
        progress: 0,
        error: `Server returned status code ${response.status}`
      };
    }
    
    return response.data;
  } catch (error) {
    console.error('Error checking analysis status:', error);
    
    // Instead of throwing, return a default status
    return {
      status: 'queued',
      progress: 0,
      error: 'Failed to check status, will retry automatically'
    };
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
    console.log("Starting uploadAndAnalyzeVideo...");
    
    // Get current session with multiple retries if needed
    let session = null;
    let retries = 0;
    const maxRetries = 3;
    
    while (!session && retries < maxRetries) {
      const { data } = await supabase.auth.getSession();
      session = data.session;
      
      if (!session) {
        console.log(`No session found, retry ${retries + 1}/${maxRetries}`);
        // Wait a moment before retrying
        await new Promise(resolve => setTimeout(resolve, 1000));
        retries++;
        
        // Try to refresh the session
        const { data: refreshData } = await supabase.auth.refreshSession();
        if (refreshData.session) {
          console.log("Session refreshed successfully");
          session = refreshData.session;
        }
      }
    }
    
    if (!session) {
      console.error("No active session found after retries");
      throw new Error("Authentication required");
    }
    
    const userId = session.user.id;
    console.log("User authenticated:", userId);
    
    // First upload to Supabase
    const response = await fetch(uri);
    const blob = await response.blob();
    
    const fileExt = fileName.split('.').pop();
    // Include user ID in the path to address RLS issue
    const filePath = `${userId}/${Date.now()}.${fileExt}`;
    
    console.log("Uploading to storage path:", filePath);
    let uploadResult;
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from('tennis-videos')
      .upload(filePath, blob);
    
    uploadResult = uploadData;
    
    if (uploadError) {
      console.error("Supabase storage upload error:", uploadError);
      
      // Check if it's an authentication error and try to refresh token
      if (uploadError.message.includes('auth') || uploadError.message.includes('401')) {
        const { data: refreshResult } = await supabase.auth.refreshSession();
        if (refreshResult.session) {
          // Retry upload with refreshed session
          console.log("Retrying upload with refreshed session");
          const { data: retryData, error: retryError } = await supabase.storage
            .from('tennis-videos')
            .upload(filePath, blob);
            
          if (retryError) {
            throw new Error(`Upload retry failed: ${retryError.message}`);
          }
          
          if (retryData) {
            uploadResult = retryData;
          }
        } else {
          throw new Error(`Upload failed: Authentication error - ${uploadError.message}`);
        }
      } else {
        throw new Error(`Upload error: ${uploadError.message}`);
      }
    }
    
    if (!uploadResult) {
      console.error("No upload data returned");
      throw new Error("Upload failed - no data returned");
    }
    
    console.log("File uploaded successfully:", uploadResult);
    const videoId = uploadResult.path;
    
    // Then submit to processing API with proper form data
    const formData = new FormData();
    formData.append('file', {
      uri: uri,
      name: fileName,
      type: `video/${fileExt}`
    } as any);
    formData.append('user_id', userId);
    formData.append('title', fileName);
    formData.append('description', '');
    
    console.log("Submitting to analysis API");
    // First try a simpler request to see if API is responding correctly
    const healthResponse = await axios.get(`${API_URL}/health`);
    console.log("API health check before upload:", healthResponse.data);
    
    // Try using direct fetch instead of axios if we're having parsing issues
    let uploadResponse;
    try {
      uploadResponse = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // Increase timeout to 5 minutes (300000ms)
        transformResponse: [(data) => {
          // Log raw response for debugging
          console.log("Raw API response:", typeof data, data.substring(0, 200) + (data.length > 200 ? "..." : ""));
          
          // Check if response looks like HTML instead of JSON
          if (typeof data === 'string' && data.trim().startsWith('<')) {
            console.log("Received HTML response instead of JSON");
            throw new Error('Server returned HTML instead of JSON. The server might be experiencing issues.');
          }
          
          // Try to parse JSON, if it fails return the raw data
          try {
            return JSON.parse(data);
          } catch (e) {
            console.log("Failed to parse JSON response:", data.substring(0, 100) + "...");
            return { error: "Invalid JSON response", raw: data.substring(0, 100) };
          }
        }]
      });
    } catch (axiosError) {
      console.log("Axios request failed, trying direct fetch as fallback");
      
      // Try with native fetch as a fallback
      const fetchResponse = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      console.log("Fetch response status:", fetchResponse.status);
      const responseText = await fetchResponse.text();
      console.log("Fetch response text:", responseText.substring(0, 200));
      
      try {
        const jsonData = JSON.parse(responseText);
        uploadResponse = { data: jsonData };
      } catch (e) {
        console.error("Failed to parse response as JSON:", e);
        throw new Error(`Server returned invalid response: ${responseText.substring(0, 100)}`);
      }
    }
    
    console.log("API response:", uploadResponse.data);
    
    // Validate the response structure
    if (uploadResponse.data.error) {
      throw new Error(`API Error: ${uploadResponse.data.error}`);
    }
    
    if (!uploadResponse.data.task_id && !uploadResponse.data.video_id) {
      console.error("Invalid API response:", uploadResponse.data);
      throw new Error("API did not return task_id or video_id");
    }
    
    return {
      videoId,
      taskId: uploadResponse.data.task_id || uploadResponse.data.video_id
    };
  } catch (error: any) {
    console.error('Error in upload and analyze:', error);
    
    // Handle specific error types
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      throw new Error('Analysis request timed out. Your video might be too large or the server is busy. Try a shorter video or try again later.');
    } else if (axios.isAxiosError(error) && !error.response) {
      // Network error
      throw new Error('Network error. Please check your internet connection and try again.');
    } else if (axios.isAxiosError(error) && error.response) {
      // Server returned an error response
      const statusCode = error.response.status;
      const message = error.response.data?.message || error.message;
      
      if (statusCode === 413) {
        throw new Error('Video file is too large. Please try uploading a shorter video.');
      } else if (statusCode >= 500) {
        throw new Error('Server error. The analysis service is currently unavailable. Please try again later.');
      } else {
        throw new Error(`Failed to upload and analyze video: ${message}`);
      }
    } else {
      // Generic error
      throw new Error(`Failed to upload and analyze video: ${error.message}`);
    }
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
    const data = response.data;
    
    // Handle different response formats
    return {
      // The API returns either status directly or inside services
      status: data.status || 'error',
      // Redis status might be in services or at the top level
      redis: data.services?.redis === 'ok' || data.redis === true,
      // Supabase status might be in services or at the top level
      supabase: data.services?.supabase === 'ok' || data.supabase === true
    };
  } catch (error) {
    console.error('API health check failed:', error);
    return {
      status: 'error',
      redis: false, 
      supabase: false
    };
  }
};