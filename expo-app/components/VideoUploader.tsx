import React, { useState } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, ActivityIndicator, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Video } from 'expo-av';
import { supabase } from '../lib/supabase';
import { decode } from 'base64-arraybuffer';
import { v4 as uuidv4 } from 'uuid';
import { useAuth } from '../hooks/useAuth';

export default function VideoUploader({ onUploadComplete }: { onUploadComplete?: (videoId: string) => void }) {
  const [video, setVideo] = useState<string | null>(null);
  const [title, setTitle] = useState('');
  const [uploading, setUploading] = useState(false);
  const { user } = useAuth();

  const pickVideo = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (permissionResult.granted === false) {
      Alert.alert('Permission Required', 'You need to allow access to your media library to upload videos.');
      return;
    }

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: true,
        quality: 1,
      });

      if (!result.canceled) {
        setVideo(result.assets[0].uri);
        // Extract filename from URI for title suggestion
        const filename = result.assets[0].uri.split('/').pop() || '';
        const suggestedTitle = filename.split('.')[0] || 'My Tennis Video';
        setTitle(suggestedTitle);
      }
    } catch (error) {
      console.error('Error picking video:', error);
      Alert.alert('Error', 'Failed to select video');
    }
  };

  const uploadVideo = async () => {
    if (!video || !user) return;
    
    try {
      setUploading(true);
      
      // Create video entry in database first
      const videoId = uuidv4();
      const videoPath = `videos/${user.id}/${videoId}.mp4`;
      
      // TODO: Implement actual video upload to Supabase Storage
      // For now we're just simulating it with a delay
      
      // This is a placeholder - in a real app you would:
      // 1. Convert video to a proper format if needed
      // 2. Upload to Supabase Storage
      // 3. Create a thumbnail
      // 4. Insert record in the database
      
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Insert video record in database
      const { error } = await supabase
        .from('videos')
        .insert({
          id: videoId,
          user_id: user.id,
          title: title || 'Untitled Video',
          video_url: `https://storage.example.com/${videoPath}`, // Placeholder URL
          analysis_status: 'pending'
        });
        
      if (error) throw error;
        
      Alert.alert('Success', 'Video uploaded successfully!');
      setVideo(null);
      setTitle('');
      
      if (onUploadComplete) {
        onUploadComplete(videoId);
      }
    } catch (error: any) {
      console.error('Error uploading video:', error);
      Alert.alert('Upload Failed', error.message || 'An unexpected error occurred');
    } finally {
      setUploading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Upload Tennis Video</Text>
      
      {video ? (
        <View style={styles.videoPreviewContainer}>
          <Video
            source={{ uri: video }}
            style={styles.videoPreview}
            useNativeControls
            resizeMode="contain"
          />
        </View>
      ) : (
        <TouchableOpacity style={styles.uploadBox} onPress={pickVideo}>
          <Text style={styles.uploadText}>Tap to select a video</Text>
        </TouchableOpacity>
      )}

      {video && (
        <TouchableOpacity 
          style={styles.uploadButton} 
          onPress={uploadVideo}
          disabled={uploading}
        >
          {uploading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Upload Video</Text>
          )}
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#0077B6',
  },
  uploadBox: {
    backgroundColor: '#F2F2F2',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#DDD',
    borderStyle: 'dashed',
    height: 200,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  uploadText: {
    color: '#888',
    fontSize: 16,
  },
  videoPreviewContainer: {
    height: 300,
    marginBottom: 20,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#000',
  },
  videoPreview: {
    width: '100%',
    height: '100%',
  },
  uploadButton: {
    backgroundColor: '#0077B6',
    borderRadius: 8,
    padding: 15,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
});