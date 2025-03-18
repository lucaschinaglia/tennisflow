import * as FileSystem from 'expo-file-system';
import { decode } from 'base64-arraybuffer';
import { v4 as uuidv4 } from 'uuid';
import { supabase } from '../lib/supabase';
import * as VideoThumbnails from 'expo-video-thumbnails';
import { Platform } from 'react-native';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';

// Cache configuration for storing temporary files
const VIDEO_CACHE_DIRECTORY = `${FileSystem.cacheDirectory}videos/`;
const THUMBNAIL_CACHE_DIRECTORY = `${FileSystem.cacheDirectory}thumbnails/`;

// Ensure cache directories exist
export const setupCacheDirectories = async () => {
  try {
    const videoDirInfo = await FileSystem.getInfoAsync(VIDEO_CACHE_DIRECTORY);
    if (!videoDirInfo.exists) {
      await FileSystem.makeDirectoryAsync(VIDEO_CACHE_DIRECTORY, { intermediates: true });
    }

    const thumbnailDirInfo = await FileSystem.getInfoAsync(THUMBNAIL_CACHE_DIRECTORY);
    if (!thumbnailDirInfo.exists) {
      await FileSystem.makeDirectoryAsync(THUMBNAIL_CACHE_DIRECTORY, { intermediates: true });
    }
  } catch (error) {
    console.error('Error setting up cache directories:', error);
    throw error;
  }
};

// Clean up cache directories
export const cleanupCache = async () => {
  try {
    await FileSystem.deleteAsync(VIDEO_CACHE_DIRECTORY, { idempotent: true });
    await FileSystem.deleteAsync(THUMBNAIL_CACHE_DIRECTORY, { idempotent: true });
    await setupCacheDirectories();
  } catch (error) {
    console.error('Error cleaning up cache:', error);
  }
};

// Compress video for more efficient upload (simulated)
// Note: In a real app, you would use a library like react-native-video-processing or FFmpeg
export const compressVideo = async (uri: string): Promise<string> => {
  try {
    // For now, this is just a simulation since we don't have actual video compression
    // In a real implementation, you would compress the video here
    
    // Copy the video to our cache directory to simulate processing
    const fileName = `compressed-${uuidv4()}.mp4`;
    const destinationUri = `${VIDEO_CACHE_DIRECTORY}${fileName}`;
    
    // Simulate compression with a delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Copy the file
    await FileSystem.copyAsync({
      from: uri,
      to: destinationUri
    });
    
    return destinationUri;
  } catch (error) {
    console.error('Error compressing video:', error);
    throw error;
  }
};

// Process and compress thumbnail
export const processThumbnail = async (uri: string): Promise<string> => {
  try {
    // Compress and resize the thumbnail for efficiency
    const processed = await manipulateAsync(
      uri,
      [{ resize: { width: 320 } }],
      { compress: 0.7, format: SaveFormat.JPEG }
    );
    
    // Save to cache
    const fileName = `thumbnail-${uuidv4()}.jpg`;
    const destinationUri = `${THUMBNAIL_CACHE_DIRECTORY}${fileName}`;
    
    await FileSystem.copyAsync({
      from: processed.uri,
      to: destinationUri
    });
    
    return destinationUri;
  } catch (error) {
    console.error('Error processing thumbnail:', error);
    throw error;
  }
};

// Generate thumbnail from video at specified time
export const generateThumbnail = async (videoUri: string, timeMs: number = 1000): Promise<string> => {
  try {
    const { uri } = await VideoThumbnails.getThumbnailAsync(videoUri, {
      time: timeMs,
      quality: 0.7,
    });
    
    return await processThumbnail(uri);
  } catch (error) {
    console.error('Error generating thumbnail:', error);
    throw error;
  }
};

// Upload file to Supabase Storage
const uploadToStorage = async (filePath: string, storagePath: string): Promise<string> => {
  try {
    // Read the file as base64
    const fileBase64 = await FileSystem.readAsStringAsync(filePath, {
      encoding: FileSystem.EncodingType.Base64,
    });
    
    // Convert to array buffer for Supabase upload
    const fileData = decode(fileBase64);
    
    // Get file extension
    const fileExt = filePath.split('.').pop();
    
    // Upload to Supabase Storage
    const { data, error } = await supabase.storage
      .from('videos')
      .upload(storagePath, fileData, {
        contentType: fileExt === 'mp4' ? 'video/mp4' : 
                    fileExt === 'jpg' ? 'image/jpeg' : 
                    'application/octet-stream',
        upsert: true
      });
    
    if (error) throw error;
    
    // Get public URL
    const { data: { publicUrl } } = supabase.storage
      .from('videos')
      .getPublicUrl(storagePath);
    
    return publicUrl;
  } catch (error) {
    console.error(`Error uploading to ${storagePath}:`, error);
    throw error;
  }
};

// Upload video and create database entry
export const uploadVideo = async (
  userId: string,
  videoUri: string,
  thumbnailUri: string,
  title: string,
  description?: string
): Promise<{ videoId: string, videoUrl: string, thumbnailUrl: string }> => {
  try {
    // First, ensure cache directories exist
    await setupCacheDirectories();
    
    // Process the video and thumbnail
    const compressedVideoUri = await compressVideo(videoUri);
    const processedThumbnailUri = await processThumbnail(thumbnailUri);
    
    // Generate unique IDs for storage paths
    const videoId = uuidv4();
    const videoPath = `${userId}/${videoId}/video.mp4`;
    const thumbnailPath = `${userId}/${videoId}/thumbnail.jpg`;
    
    // Upload to Supabase Storage
    const videoUrl = await uploadToStorage(compressedVideoUri, videoPath);
    const thumbnailUrl = await uploadToStorage(processedThumbnailUri, thumbnailPath);
    
    // Create entry in the videos table
    const { data, error } = await supabase
      .from('videos')
      .insert({
        id: videoId,
        user_id: userId,
        title,
        description,
        video_url: videoUrl,
        thumbnail_url: thumbnailUrl,
        analysis_status: 'pending'
      })
      .select()
      .single();
    
    if (error) throw error;
    
    // Clean up temporary files
    await FileSystem.deleteAsync(compressedVideoUri, { idempotent: true });
    await FileSystem.deleteAsync(processedThumbnailUri, { idempotent: true });
    
    return {
      videoId,
      videoUrl,
      thumbnailUrl
    };
  } catch (error) {
    console.error('Error uploading video:', error);
    throw error;
  }
};

// Request video analysis from the ML service
export const requestVideoAnalysis = async (videoId: string): Promise<void> => {
  try {
    // In a real implementation, this would make a request to your ML API
    // For now, we'll update the status directly in the database
    
    // Update the video status to "processing"
    const { error } = await supabase
      .from('videos')
      .update({ analysis_status: 'processing' })
      .eq('id', videoId);
    
    if (error) throw error;
    
    // Simulate an API call to the ML service
    console.log(`Requesting analysis for video ${videoId}`);
    
    // In a real app, this would be replaced with an actual API call
    // to trigger the ML processing service
  } catch (error) {
    console.error('Error requesting video analysis:', error);
    throw error;
  }
};

// Get video details
export const getVideoDetails = async (videoId: string): Promise<any> => {
  try {
    const { data, error } = await supabase
      .from('videos')
      .select('*')
      .eq('id', videoId)
      .single();
    
    if (error) throw error;
    
    return data;
  } catch (error) {
    console.error('Error fetching video details:', error);
    throw error;
  }
};

// Get user videos
export const getUserVideos = async (userId: string): Promise<any[]> => {
  try {
    const { data, error } = await supabase
      .from('videos')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });
    
    if (error) throw error;
    
    return data || [];
  } catch (error) {
    console.error('Error fetching user videos:', error);
    throw error;
  }
};

// Delete video
export const deleteVideo = async (videoId: string, userId: string): Promise<void> => {
  try {
    // First get the video to check ownership and get paths
    const { data, error } = await supabase
      .from('videos')
      .select('*')
      .eq('id', videoId)
      .eq('user_id', userId)
      .single();
    
    if (error) throw error;
    if (!data) throw new Error('Video not found or you do not have permission to delete it');
    
    // Delete from storage
    const storagePath = `${userId}/${videoId}`;
    const { error: storageError } = await supabase.storage
      .from('videos')
      .remove([`${storagePath}/video.mp4`, `${storagePath}/thumbnail.jpg`]);
    
    if (storageError) console.error('Error deleting from storage:', storageError);
    
    // Delete from database
    const { error: dbError } = await supabase
      .from('videos')
      .delete()
      .eq('id', videoId);
    
    if (dbError) throw dbError;
  } catch (error) {
    console.error('Error deleting video:', error);
    throw error;
  }
};