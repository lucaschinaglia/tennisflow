import { supabase } from '../lib/supabase';

// Types for analysis results
export type SwingMetrics = {
  racketSpeed: number;        // mph
  hipRotation: number;        // degrees
  shoulderRotation: number;   // degrees
  kneeFlexion: number;        // degrees
  weightTransfer: number;     // percentage
  balanceScore: number;       // 0-100
  followThrough: number;      // 0-100
  consistency: number;        // 0-100
};

export type SwingPhase = 
  | 'preparation' 
  | 'backswing' 
  | 'forward-swing' 
  | 'contact' 
  | 'follow-through';

export type FrameAnalysis = {
  frameNumber: number;
  timestamp: number;
  poseData: {
    keypoints: {
      name: string;
      position: { x: number; y: number };
      confidence: number;
    }[];
    connections: {
      from: string;
      to: string;
    }[];
  };
  swingPhase: SwingPhase | null;
  annotations: {
    type: 'angle' | 'position' | 'movement' | 'warning';
    text: string;
    position: { x: number; y: number };
    value?: number;
    color?: string;
  }[];
};

export type VideoAnalysisResults = {
  videoId: string;
  duration: number;
  frames: FrameAnalysis[];
  summary: {
    swingType: 'forehand' | 'backhand' | 'serve' | 'volley' | 'unknown';
    swingCount: number;
    averageMetrics: SwingMetrics;
    strengths: string[];
    weaknesses: string[];
    improvementSuggestions: string[];
  };
  swings: {
    id: string;
    startTime: number;
    endTime: number;
    swingType: 'forehand' | 'backhand' | 'serve' | 'volley' | 'unknown';
    metrics: SwingMetrics;
    keyFrames: number[];
    score: number;  // 0-100
  }[];
};

// Fetch analysis results for a video
export const getVideoAnalysis = async (videoId: string): Promise<VideoAnalysisResults | null> => {
  try {
    // Get the video details including analysis_results
    const { data: video, error } = await supabase
      .from('videos')
      .select('*')
      .eq('id', videoId)
      .single();
    
    if (error) throw error;
    if (!video) throw new Error('Video not found');
    
    // Return the analysis results if available and completed
    if (video.analysis_status === 'completed' && video.analysis_results) {
      return video.analysis_results as VideoAnalysisResults;
    }
    
    return null;
  } catch (error) {
    console.error('Error fetching video analysis:', error);
    throw error;
  }
};

// Fetch detailed analysis for specific frames
export const getFrameAnalysisDetails = async (videoId: string, frameNumbers: number[]): Promise<FrameAnalysis[]> => {
  try {
    // Get the frame analysis details
    const { data, error } = await supabase
      .from('analysis_details')
      .select('*')
      .eq('video_id', videoId)
      .in('frame_number', frameNumbers)
      .order('frame_number', { ascending: true });
    
    if (error) throw error;
    
    return data.map(item => ({
      frameNumber: item.frame_number,
      timestamp: item.timestamp,
      poseData: item.pose_data,
      swingPhase: item.swing_phase,
      annotations: item.annotations
    }));
  } catch (error) {
    console.error('Error fetching frame analysis details:', error);
    throw error;
  }
};

// Get the analysis status of a video
export const getAnalysisStatus = async (videoId: string): Promise<string> => {
  try {
    const { data, error } = await supabase
      .from('videos')
      .select('analysis_status')
      .eq('id', videoId)
      .single();
    
    if (error) throw error;
    
    return data.analysis_status;
  } catch (error) {
    console.error('Error fetching analysis status:', error);
    throw error;
  }
};

// Request a new analysis for a video (typically for re-analysis)
export const requestAnalysis = async (videoId: string): Promise<void> => {
  try {
    // First, update the status to 'pending'
    const { error } = await supabase
      .from('videos')
      .update({ analysis_status: 'pending' })
      .eq('id', videoId);
    
    if (error) throw error;
    
    // In a real implementation, this would make an API call to your
    // ML service to trigger the analysis process. For now, we'll
    // simulate this with a status change.
    
    // This would likely be a call to an external API, e.g.:
    // const response = await fetch(`${API_BASE_URL}/analyze`, {
    //   method: 'POST',
    //   headers: {
    //     'Content-Type': 'application/json',
    //     'Authorization': `Bearer ${API_KEY}`
    //   },
    //   body: JSON.stringify({ videoId })
    // });
    
    // if (!response.ok) {
    //   throw new Error('Failed to request analysis');
    // }
    
    console.log(`Requested analysis for video ${videoId}`);
  } catch (error) {
    console.error('Error requesting analysis:', error);
    throw error;
  }
};

// Get feedback based on analysis results
export const getFeedback = async (videoId: string): Promise<any[]> => {
  try {
    const { data, error } = await supabase
      .from('feedback')
      .select('*')
      .eq('video_id', videoId)
      .order('created_at', { ascending: true });
    
    if (error) throw error;
    
    return data || [];
  } catch (error) {
    console.error('Error fetching feedback:', error);
    throw error;
  }
};

// Add user feedback for a video
export const addFeedback = async (
  videoId: string, 
  userId: string, 
  content: string, 
  timestamp?: number, 
  isPublic: boolean = false
): Promise<void> => {
  try {
    const { error } = await supabase
      .from('feedback')
      .insert({
        video_id: videoId,
        user_id: userId,
        feedback_type: 'user',
        content,
        timestamp,
        is_public: isPublic
      });
    
    if (error) throw error;
  } catch (error) {
    console.error('Error adding feedback:', error);
    throw error;
  }
};

// Get improvement metrics over time
export const getProgressMetrics = async (userId: string, metricName?: string): Promise<any[]> => {
  try {
    let query = supabase
      .from('progress')
      .select('*,videos(created_at)')
      .eq('user_id', userId)
      .order('created_at', { ascending: true });
    
    if (metricName) {
      query = query.eq('metric_name', metricName);
    }
    
    const { data, error } = await query;
    
    if (error) throw error;
    
    return data || [];
  } catch (error) {
    console.error('Error fetching progress metrics:', error);
    throw error;
  }
};

// Compare two videos (e.g., before/after training)
export const compareVideos = async (videoId1: string, videoId2: string): Promise<any> => {
  try {
    // In a real implementation, this would call an API to compare the videos
    // or retrieve stored comparison data. For now, we'll fake it with
    // direct database queries.
    
    const [analysis1, analysis2] = await Promise.all([
      getVideoAnalysis(videoId1),
      getVideoAnalysis(videoId2)
    ]);
    
    if (!analysis1 || !analysis2) {
      throw new Error('Analysis not available for one or both videos');
    }
    
    // Calculate differences in metrics
    const metricChanges = Object.keys(analysis1.summary.averageMetrics).reduce((acc, key) => {
      const metric = key as keyof SwingMetrics;
      const value1 = analysis1.summary.averageMetrics[metric];
      const value2 = analysis2.summary.averageMetrics[metric];
      const percentChange = ((value2 - value1) / value1) * 100;
      
      acc[metric] = {
        before: value1,
        after: value2,
        difference: value2 - value1,
        percentChange
      };
      
      return acc;
    }, {} as Record<string, any>);
    
    return {
      video1: {
        id: videoId1,
        analysis: analysis1
      },
      video2: {
        id: videoId2,
        analysis: analysis2
      },
      comparison: {
        metricChanges,
        improvedAreas: Object.keys(metricChanges).filter(key => metricChanges[key].percentChange > 0),
        declinedAreas: Object.keys(metricChanges).filter(key => metricChanges[key].percentChange < 0)
      }
    };
  } catch (error) {
    console.error('Error comparing videos:', error);
    throw error;
  }
};