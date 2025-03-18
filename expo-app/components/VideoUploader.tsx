import React, { useState, useEffect } from 'react';
import { 
  StyleSheet, 
  View, 
  Text, 
  TouchableOpacity, 
  ActivityIndicator, 
  Alert,
  TextInput,
  ScrollView,
  Modal
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Video } from 'expo-av';
import { useAuth } from '../hooks/useAuth';
import { Ionicons } from '@expo/vector-icons';
import * as VideoService from '../services/videoService';
import CameraScreen from '../screens/CameraScreen';
import VideoEditorScreen from '../screens/VideoEditorScreen';

type VideoUploaderProps = {
  onUploadComplete?: (videoId: string) => void;
  onCancel?: () => void;
};

export default function VideoUploader({ onUploadComplete, onCancel }: VideoUploaderProps) {
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [thumbnailUri, setThumbnailUri] = useState<string | null>(null);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [uploading, setUploading] = useState(false);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [showCamera, setShowCamera] = useState(false);
  const [showEditor, setShowEditor] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const { user } = useAuth();

  useEffect(() => {
    // Setup cache directories when component mounts
    VideoService.setupCacheDirectories();
    
    // Clean up cache when component unmounts
    return () => {
      VideoService.cleanupCache();
    };
  }, []);

  const openCamera = () => {
    setShowCamera(true);
  };

  const openImagePicker = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (permissionResult.granted === false) {
      Alert.alert('Permission Required', 'You need to allow access to your media library to upload videos.');
      return;
    }

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: false,
        quality: 1,
      });

      if (!result.canceled) {
        // Get the selected video
        const selectedUri = result.assets[0].uri;
        setVideoUri(selectedUri);
        
        // Generate a thumbnail
        const thumbnail = await VideoService.generateThumbnail(selectedUri);
        setThumbnailUri(thumbnail);
        
        // Extract filename for title suggestion
        const filename = selectedUri.split('/').pop() || '';
        const suggestedTitle = filename.split('.')[0] || 'My Tennis Video';
        setTitle(suggestedTitle);
        
        // Open editor
        setShowEditor(true);
      }
    } catch (error) {
      console.error('Error picking video:', error);
      Alert.alert('Error', 'Failed to select video');
    }
  };

  const handleCameraClose = () => {
    setShowCamera(false);
  };

  const handleRecordingComplete = async (uri: string) => {
    try {
      setShowCamera(false);
      setVideoUri(uri);
      
      // Generate a thumbnail
      const thumbnail = await VideoService.generateThumbnail(uri);
      setThumbnailUri(thumbnail);
      
      // Set a default title
      const date = new Date();
      setTitle(`Tennis Video ${date.toLocaleDateString()}`);
      
      // Open editor
      setShowEditor(true);
    } catch (error) {
      console.error('Error processing recording:', error);
      Alert.alert('Error', 'Failed to process recorded video');
    }
  };

  const handleEditorCancel = () => {
    setShowEditor(false);
  };

  const handleEditorSave = (uri: string, thumbnail: string, duration: number) => {
    setVideoUri(uri);
    setThumbnailUri(thumbnail);
    setVideoDuration(duration);
    setShowEditor(false);
  };

  const uploadVideo = async () => {
    if (!videoUri || !thumbnailUri || !user) {
      Alert.alert('Error', 'Missing video, thumbnail, or user information');
      return;
    }
    
    if (!title.trim()) {
      Alert.alert('Title Required', 'Please enter a title for your video');
      return;
    }
    
    try {
      setUploading(true);
      setUploadProgress(0);
      
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 95) {
            clearInterval(progressInterval);
            return 95;
          }
          return prev + 5;
        });
      }, 300);
      
      // Upload video
      const { videoId } = await VideoService.uploadVideo(
        user.id,
        videoUri,
        thumbnailUri,
        title,
        description
      );
      
      // Clear progress interval
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      // Request video analysis
      await VideoService.requestVideoAnalysis(videoId);
      
      Alert.alert(
        'Upload Complete', 
        'Your video has been uploaded and is being processed for analysis. You will be notified when the analysis is complete.'
      );
      
      // Reset form
      setVideoUri(null);
      setThumbnailUri(null);
      setTitle('');
      setDescription('');
      
      // Call the callback if provided
      if (onUploadComplete) {
        onUploadComplete(videoId);
      }
    } catch (error: any) {
      console.error('Error uploading video:', error);
      Alert.alert('Upload Failed', error.message || 'An unexpected error occurred');
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <View style={styles.container}>
      {/* Camera Modal */}
      <Modal 
        visible={showCamera}
        animationType="slide"
        presentationStyle="fullScreen"
      >
        <CameraScreen 
          onRecordingComplete={handleRecordingComplete}
          onClose={handleCameraClose}
        />
      </Modal>
      
      {/* Video Editor Modal */}
      <Modal 
        visible={showEditor && !!videoUri}
        animationType="slide"
        presentationStyle="fullScreen"
      >
        {videoUri && (
          <VideoEditorScreen 
            videoUri={videoUri}
            onSave={handleEditorSave}
            onCancel={handleEditorCancel}
          />
        )}
      </Modal>
      
      <ScrollView style={styles.scrollView}>
        <Text style={styles.title}>Upload Tennis Video</Text>
        
        {videoUri && thumbnailUri ? (
          <View style={styles.videoPreviewContainer}>
            <Video
              source={{ uri: videoUri }}
              style={styles.videoPreview}
              useNativeControls
              resizeMode="contain"
            />
            
            <TouchableOpacity 
              style={styles.changeVideoButton} 
              onPress={() => {
                setVideoUri(null);
                setThumbnailUri(null);
              }}
              disabled={uploading}
            >
              <Text style={styles.changeButtonText}>Change Video</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <View style={styles.uploadOptions}>
            <TouchableOpacity 
              style={styles.uploadOption} 
              onPress={openCamera}
            >
              <Ionicons name="camera-outline" size={40} color="#0077B6" />
              <Text style={styles.uploadOptionText}>Record Video</Text>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={styles.uploadOption} 
              onPress={openImagePicker}
            >
              <Ionicons name="images-outline" size={40} color="#0077B6" />
              <Text style={styles.uploadOptionText}>Choose from Library</Text>
            </TouchableOpacity>
          </View>
        )}
        
        {videoUri && (
          <View style={styles.formContainer}>
            <Text style={styles.label}>Title</Text>
            <TextInput
              style={styles.input}
              value={title}
              onChangeText={setTitle}
              placeholder="Enter a title for your video"
              maxLength={100}
              editable={!uploading}
            />
            
            <Text style={styles.label}>Description (optional)</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              value={description}
              onChangeText={setDescription}
              placeholder="Add a description for your tennis video"
              multiline
              numberOfLines={4}
              maxLength={500}
              editable={!uploading}
            />
            
            <TouchableOpacity 
              style={[styles.uploadButton, uploading && styles.uploadingButton]} 
              onPress={uploadVideo}
              disabled={uploading}
            >
              {uploading ? (
                <View style={styles.progressContainer}>
                  <ActivityIndicator color="#fff" size="small" />
                  <Text style={styles.progressText}>{uploadProgress}%</Text>
                </View>
              ) : (
                <Text style={styles.buttonText}>Upload Video</Text>
              )}
            </TouchableOpacity>
            
            {!uploading && onCancel && (
              <TouchableOpacity 
                style={styles.cancelButton} 
                onPress={onCancel}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
            )}
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  scrollView: {
    flex: 1,
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#0077B6',
  },
  uploadOptions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginVertical: 30,
  },
  uploadOption: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    backgroundColor: '#F2F2F2',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#DDD',
    borderStyle: 'dashed',
    width: '45%',
    height: 150,
  },
  uploadOptionText: {
    marginTop: 10,
    fontSize: 16,
    color: '#0077B6',
    textAlign: 'center',
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
  changeVideoButton: {
    position: 'absolute',
    bottom: 10,
    right: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  changeButtonText: {
    color: 'white',
    fontSize: 14,
  },
  formContainer: {
    marginTop: 10,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#333',
  },
  input: {
    backgroundColor: '#F2F2F2',
    borderRadius: 8,
    padding: 15,
    marginBottom: 20,
    fontSize: 16,
  },
  textArea: {
    height: 100,
    textAlignVertical: 'top',
  },
  uploadButton: {
    backgroundColor: '#0077B6',
    borderRadius: 8,
    padding: 15,
    alignItems: 'center',
    marginTop: 10,
  },
  uploadingButton: {
    backgroundColor: '#0077B6',
    opacity: 0.8,
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  progressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  progressText: {
    color: '#fff',
    marginLeft: 10,
    fontWeight: 'bold',
    fontSize: 16,
  },
  cancelButton: {
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    padding: 15,
    alignItems: 'center',
    marginTop: 10,
  },
  cancelButtonText: {
    color: '#333',
    fontWeight: 'bold',
    fontSize: 16,
  },
});