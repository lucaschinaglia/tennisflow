// Supabase Edge Function for Tennis Video Analysis
// This function processes uploaded videos using OpenPose/MediaPipe for pose estimation

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

// Initialize Supabase client
const supabaseUrl = Deno.env.get('SUPABASE_URL') || ''
const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || ''
const supabase = createClient(supabaseUrl, supabaseServiceKey)

// Set up OpenCV.js and MediaPipe instance
// In a real app, you'd use OpenPose here but for edge functions we'll simulate with MediaPipe which is more JS-friendly

// Process function that analyzes video frames to identify tennis poses
async function processVideo(videoId: string, videoUrl: string): Promise<any> {
  console.log(`Processing video ${videoId} at ${videoUrl}`)
  
  try {
    // Step 1: Update status to 'processing'
    await updateVideoStatus(videoId, 'processing')
    
    // Step 2: Download video from storage for processing
    // This would hit our OpenPose processing server in production
    
    // Step 3: Extract frames from video
    // In production, this would use OpenCV/FFmpeg
    
    // Step 4: Process frames with OpenPose/MediaPipe
    // For demonstration, we'll simulate results
    const results = await simulateVideoAnalysis(videoId)
    
    // Step 5: Update database with results
    await updateAnalysisResults(videoId, results)
    
    // Step 6: Update status to 'completed'
    await updateVideoStatus(videoId, 'completed')
    
    return { success: true, videoId, status: 'completed' }
  } catch (error) {
    console.error(`Error processing video ${videoId}:`, error)
    
    // Update status to 'failed'
    await updateVideoStatus(videoId, 'failed')
    
    return { success: false, videoId, status: 'failed', error: error.message }
  }
}

// Update the video status in the database
async function updateVideoStatus(videoId: string, status: string): Promise<void> {
  const { error } = await supabase
    .from('videos')
    .update({ analysis_status: status })
    .eq('id', videoId)
  
  if (error) throw error
}

// Update the analysis results in the database
async function updateAnalysisResults(videoId: string, results: any): Promise<void> {
  const { error } = await supabase
    .from('videos')
    .update({ 
      analysis_results: results,
      analysis_status: 'completed'
    })
    .eq('id', videoId)
  
  if (error) throw error
  
  // Insert detailed analysis into the analysis_details table
  if (results.frames && results.frames.length > 0) {
    const detailsToInsert = results.frames.map((frame: any) => ({
      video_id: videoId,
      frame_number: frame.frameNumber,
      timestamp: frame.timestamp,
      pose_data: frame.poseData,
      swing_phase: frame.swingPhase,
      annotations: frame.annotations
    }))
    
    const { error: detailsError } = await supabase
      .from('analysis_details')
      .insert(detailsToInsert)
    
    if (detailsError) throw detailsError
  }
}

// Simulate video analysis for demonstration purposes
// In production, this would be replaced with actual OpenPose/MediaPipe processing
async function simulateVideoAnalysis(videoId: string): Promise<any> {
  // In a real implementation, this would be the result of OpenPose processing
  // For now we'll return a simulated result structure
  
  const NUM_FRAMES = 60 // Simulate 60 frames (2 seconds at 30fps)
  
  const frames = Array.from({ length: NUM_FRAMES }, (_, i) => {
    const frameNumber = i
    const timestamp = i / 30 // Assuming 30fps
    
    // Determine swing phase based on frame position
    let swingPhase = null
    if (i < 10) swingPhase = 'preparation'
    else if (i < 20) swingPhase = 'backswing'
    else if (i < 30) swingPhase = 'forward-swing'
    else if (i < 35) swingPhase = 'contact'
    else swingPhase = 'follow-through'
    
    // Simulate pose keypoints (simplified)
    const keypoints = [
      { name: 'nose', position: { x: 0.5, y: 0.2 }, confidence: 0.95 },
      { name: 'left_shoulder', position: { x: 0.4, y: 0.3 }, confidence: 0.9 },
      { name: 'right_shoulder', position: { x: 0.6, y: 0.3 }, confidence: 0.9 },
      { name: 'left_elbow', position: { x: 0.3, y: 0.4 }, confidence: 0.85 },
      { name: 'right_elbow', position: { x: 0.7, y: 0.4 }, confidence: 0.85 },
      { name: 'left_wrist', position: { x: 0.2, y: 0.5 }, confidence: 0.8 },
      { name: 'right_wrist', position: { x: 0.8, y: 0.5 }, confidence: 0.8 },
      { name: 'left_hip', position: { x: 0.45, y: 0.6 }, confidence: 0.9 },
      { name: 'right_hip', position: { x: 0.55, y: 0.6 }, confidence: 0.9 },
      { name: 'left_knee', position: { x: 0.4, y: 0.75 }, confidence: 0.85 },
      { name: 'right_knee', position: { x: 0.6, y: 0.75 }, confidence: 0.85 },
      { name: 'left_ankle', position: { x: 0.35, y: 0.9 }, confidence: 0.8 },
      { name: 'right_ankle', position: { x: 0.65, y: 0.9 }, confidence: 0.8 },
    ]
    
    // Add some motion to the keypoints based on swing phase
    keypoints.forEach(kp => {
      // Add slight variations based on frame number and swing phase
      if (swingPhase === 'backswing') {
        if (kp.name.includes('right')) {
          kp.position.x += 0.05 * (i - 10) / 10
          kp.position.y -= 0.03 * (i - 10) / 10
        }
      } else if (swingPhase === 'forward-swing') {
        if (kp.name.includes('right')) {
          kp.position.x -= 0.08 * (i - 20) / 10
          kp.position.y += 0.01 * (i - 20) / 10
        }
      }
      
      // Ensure positions stay in valid range (0-1)
      kp.position.x = Math.max(0, Math.min(1, kp.position.x))
      kp.position.y = Math.max(0, Math.min(1, kp.position.y))
    })
    
    // Simulate annotations based on swing phase
    const annotations = []
    
    if (swingPhase === 'backswing') {
      annotations.push({
        type: 'angle',
        text: 'Hip Rotation: 45°',
        position: { x: 0.5, y: 0.6 },
        value: 45,
        color: '#4CAF50'
      })
    } else if (swingPhase === 'contact') {
      annotations.push({
        type: 'angle',
        text: 'Elbow Angle: 160°',
        position: { x: 0.7, y: 0.4 },
        value: 160,
        color: '#FFC107'
      })
    } else if (swingPhase === 'follow-through') {
      annotations.push({
        type: 'position',
        text: 'Finish Position',
        position: { x: 0.8, y: 0.5 },
        color: '#2196F3'
      })
    }
    
    return {
      frameNumber,
      timestamp,
      poseData: {
        keypoints,
        connections: [
          { from: 'left_shoulder', to: 'right_shoulder' },
          { from: 'left_shoulder', to: 'left_elbow' },
          { from: 'right_shoulder', to: 'right_elbow' },
          { from: 'left_elbow', to: 'left_wrist' },
          { from: 'right_elbow', to: 'right_wrist' },
          { from: 'left_shoulder', to: 'left_hip' },
          { from: 'right_shoulder', to: 'right_hip' },
          { from: 'left_hip', to: 'right_hip' },
          { from: 'left_hip', to: 'left_knee' },
          { from: 'right_hip', to: 'right_knee' },
          { from: 'left_knee', to: 'left_ankle' },
          { from: 'right_knee', to: 'right_ankle' },
        ]
      },
      swingPhase,
      annotations
    }
  })
  
  // Find key frames for each swing phase
  const keyFrames = [5, 15, 25, 33, 45]
  
  // Calculate metrics based on poses
  const metrics = {
    racketSpeed: 85 + Math.random() * 10,
    hipRotation: 45 + Math.random() * 10,
    shoulderRotation: 80 + Math.random() * 15,
    kneeFlexion: 30 + Math.random() * 10,
    weightTransfer: 75 + Math.random() * 15,
    balanceScore: 65 + Math.random() * 20,
    followThrough: 70 + Math.random() * 20,
    consistency: 60 + Math.random() * 30
  }
  
  // Generate a full analysis report
  return {
    videoId,
    duration: NUM_FRAMES / 30, // Duration in seconds
    frames,
    summary: {
      swingType: 'forehand',
      swingCount: 1,
      averageMetrics: metrics,
      strengths: [
        'Good follow-through extension',
        'Proper hip rotation during backswing',
        'Stable head position throughout the swing'
      ],
      weaknesses: [
        'Inconsistent knee bend during preparation',
        'Could improve weight transfer timing',
        'Racket face slightly open at contact'
      ],
      improvementSuggestions: [
        'Focus on deeper knee bend during preparation phase',
        'Practice weight transfer drills to improve timing',
        'Work on contact point positioning for better racket face control'
      ]
    },
    swings: [
      {
        id: crypto.randomUUID(),
        startTime: 0,
        endTime: NUM_FRAMES / 30,
        swingType: 'forehand',
        metrics,
        keyFrames,
        score: 75 + Math.random() * 15
      }
    ]
  }
}

// Main handler for the Supabase Edge Function
serve(async (req) => {
  // CORS headers
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  }

  // Handle CORS preflight request
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers })
  }

  try {
    // Extract request data
    const { videoId } = await req.json()
    
    if (!videoId) {
      return new Response(
        JSON.stringify({ error: 'Missing videoId parameter' }),
        { headers, status: 400 }
      )
    }
    
    // Get the video details from the database
    const { data: video, error } = await supabase
      .from('videos')
      .select('*')
      .eq('id', videoId)
      .single()
    
    if (error || !video) {
      return new Response(
        JSON.stringify({ error: 'Video not found' }),
        { headers, status: 404 }
      )
    }
    
    // Process the video
    const result = await processVideo(videoId, video.video_url)
    
    return new Response(
      JSON.stringify(result),
      { headers, status: 200 }
    )
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { headers, status: 500 }
    )
  }
})