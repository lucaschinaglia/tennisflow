import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Image,
  Platform,
  SafeAreaView,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { CameraView, CameraType as ExpoCameraType, useCameraPermissions, VideoQuality } from 'expo-camera';
import { Video, ResizeMode } from 'expo-av';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { uploadAndAnalyzeVideo, checkApiHealth } from '../services/analysisService';
import { CameraType, RootStackParamList } from '../types';

type VideoUploaderScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'VideoUploader'>;

const VideoUploader: React.FC = () => {
  const navigation = useNavigation<VideoUploaderScreenNavigationProp>();
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraType, setCameraType] = useState<CameraType>(CameraType.back);
  const [isRecording, setIsRecording] = useState(false);
  const [video, setVideo] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const cameraRef = useRef<CameraView>(null);
  const [isHealthy, setIsHealthy] = useState(true);

  useEffect(() => {
    (async () => {
      if (!permission?.granted) {
        await requestPermission();
      }
    })();
  }, [permission, requestPermission]);

  const startRecording = async () => {
    if (cameraRef.current) {
      setIsRecording(true);
      try {
        const videoRecordPromise = cameraRef.current.recordAsync({
          maxDuration: 15
        });
        
        const data = await videoRecordPromise;
        if (data) {
          setVideo(data.uri);
        }
      } catch (error) {
        console.error('Error recording video:', error);
        Alert.alert('Error', 'Failed to record video');
      }
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (cameraRef.current && isRecording) {
      cameraRef.current.stopRecording();
      setIsRecording(false);
    }
  };

  const pickVideo = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      allowsEditing: false,
      quality: 1,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setVideo(result.assets[0].uri);
    }
  };

  const checkHealth = async () => {
    try {
      setIsHealthy(false);
      const healthStatus = await checkApiHealth();
      console.log('API health status:', healthStatus);
      
      // API is considered healthy if the status is "healthy" or "ok"
      setIsHealthy(healthStatus.status === 'healthy' || healthStatus.status === 'ok');
    } catch (error) {
      console.error('Error checking API health:', error);
      setIsHealthy(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  const uploadVideo = async () => {
    if (!video) return;
    
    if (!isHealthy) {
      Alert.alert(
        'Service Unavailable', 
        'The analysis service is not available. Please check your connection and try again later.',
        [{ text: 'OK' }]
      );
      return;
    }

    try {
      setUploading(true);
      setUploadProgress(0);
      
      // Extract filename from URI
      const uriParts = video.split('/');
      const fileName = uriParts[uriParts.length - 1];
      
      // Show a message for large videos
      if (Platform.OS === 'ios') {
        // Alert user that analysis might take time
        Alert.alert(
          'Analysis in Progress',
          'Your video is being uploaded and analyzed. This may take a few minutes. You will be notified when complete.',
          [{ text: 'OK' }]
        );
      }
      
      // Upload the video
      const result = await uploadAndAnalyzeVideo(video, fileName);
      
      setUploading(false);
      navigation.navigate('VideoAnalysis', { 
        videoId: result.videoId,
        taskId: result.taskId,
        videoUri: video
      });
    } catch (error: any) {
      setUploading(false);
      console.error('Error uploading video:', error);
      
      // Show a more specific error message based on the error type
      Alert.alert(
        'Upload Failed', 
        error.message || 'There was an error uploading your video. Please try again.',
        [{ text: 'OK' }]
      );
    }
  };

  const resetVideo = () => {
    setVideo(null);
  };

  const toggleCameraType = () => {
    setCameraType(prevType => 
      prevType === CameraType.back ? CameraType.front : CameraType.back
    );
  };

  if (!permission?.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>No access to camera</Text>
        <TouchableOpacity 
          style={styles.button}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.buttonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />
      
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="chevron-back" size={24} color="#FFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>
          {video ? 'Review Video' : 'Record Swing'}
        </Text>
        <View style={{ width: 40 }} />
      </View>

      {!permission?.granted ? (
        <View style={styles.permissionContainer}>
          <Text style={styles.permissionText}>We need camera permission to record your swing</Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Text style={styles.permissionButtonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      ) : !video ? (
        // Camera View
        <View style={styles.cameraContainer}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={cameraType === CameraType.back ? 'back' : 'front'}
            ratio="16:9"
          >
            <View style={styles.cameraControls}>
              <TouchableOpacity
                style={styles.flipButton}
                onPress={toggleCameraType}
              >
                <Ionicons name="camera-reverse" size={30} color="#FFF" />
              </TouchableOpacity>
            </View>
          </CameraView>
          
          <View style={styles.controls}>
            <TouchableOpacity
              style={styles.galleryButton}
              onPress={pickVideo}
            >
              <Ionicons name="images" size={30} color="#FFF" />
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[
                styles.recordButton,
                isRecording && styles.recordingButton
              ]}
              onPress={isRecording ? stopRecording : startRecording}
            >
              {isRecording ? (
                <View style={styles.stopRecordingIcon} />
              ) : (
                <View style={styles.startRecordingIcon} />
              )}
            </TouchableOpacity>
            
            <View style={{ width: 50 }} />
          </View>
          
          <Text style={styles.instructions}>
            Position yourself to capture your full swing
          </Text>
        </View>
      ) : (
        // Video Preview
        <View style={styles.previewContainer}>
          <Video
            source={{ uri: video }}
            style={styles.videoPreview}
            useNativeControls
            resizeMode={ResizeMode.CONTAIN}
            isLooping
          />
          
          <View style={styles.previewControls}>
            <TouchableOpacity
              style={styles.previewButton}
              onPress={resetVideo}
            >
              <Ionicons name="refresh" size={24} color="#FFF" />
              <Text style={styles.previewButtonText}>Retake</Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[styles.previewButton, styles.uploadButton]}
              onPress={uploadVideo}
              disabled={uploading}
            >
              {uploading ? (
                <ActivityIndicator size="small" color="#FFF" />
              ) : (
                <>
                  <Ionicons name="cloud-upload" size={24} color="#FFF" />
                  <Text style={styles.previewButtonText}>Analyze</Text>
                </>
              )}
            </TouchableOpacity>
          </View>
          
          {uploading && (
            <View style={styles.progressContainer}>
              <View 
                style={[
                  styles.progressBar, 
                  { width: `${uploadProgress}%` }
                ]} 
              />
              <Text style={styles.progressText}>
                Uploading... {Math.round(uploadProgress)}%
              </Text>
            </View>
          )}
        </View>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  cameraContainer: {
    flex: 1,
    position: 'relative',
  },
  camera: {
    flex: 1,
  },
  cameraControls: {
    position: 'absolute',
    top: 16,
    right: 16,
  },
  flipButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: 30,
    padding: 10,
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    position: 'absolute',
    bottom: 60,
    left: 0,
    right: 0,
  },
  recordButton: {
    borderWidth: 3,
    borderColor: '#FFF',
    borderRadius: 40,
    height: 80,
    width: 80,
    justifyContent: 'center',
    alignItems: 'center',
  },
  recordingButton: {
    backgroundColor: '#FF4136',
    borderColor: '#FFF',
  },
  startRecordingIcon: {
    backgroundColor: '#FF4136',
    width: 64,
    height: 64,
    borderRadius: 32,
  },
  stopRecordingIcon: {
    backgroundColor: '#FFF',
    width: 30,
    height: 30,
    borderRadius: 5,
  },
  galleryButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: 30,
    padding: 10,
  },
  instructions: {
    position: 'absolute',
    bottom: 20,
    left: 0,
    right: 0,
    textAlign: 'center',
    color: '#FFF',
    fontSize: 14,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    paddingVertical: 8,
  },
  previewContainer: {
    flex: 1,
    position: 'relative',
  },
  videoPreview: {
    flex: 1,
  },
  previewControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
  },
  previewButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    padding: 12,
    borderRadius: 25,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  uploadButton: {
    backgroundColor: '#2196F3',
  },
  previewButtonText: {
    color: '#FFF',
    marginLeft: 8,
    fontSize: 16,
    fontWeight: 'bold',
  },
  progressContainer: {
    position: 'absolute',
    bottom: 100,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    borderRadius: 10,
    padding: 10,
  },
  progressBar: {
    height: 4,
    backgroundColor: '#2196F3',
    borderRadius: 2,
    marginBottom: 8,
  },
  progressText: {
    color: '#FFF',
    fontSize: 12,
    textAlign: 'center',
  },
  text: {
    color: 'white',
    fontSize: 16,
    textAlign: 'center',
  },
  button: {
    backgroundColor: '#2196F3',
    padding: 12,
    borderRadius: 8,
    marginTop: 20,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    textAlign: 'center',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  permissionText: {
    color: '#FFF',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
  },
  permissionButton: {
    backgroundColor: '#2196F3',
    padding: 12,
    borderRadius: 8,
  },
  permissionButtonText: {
    color: 'white',
    fontSize: 16,
    textAlign: 'center',
  },
});

export default VideoUploader;