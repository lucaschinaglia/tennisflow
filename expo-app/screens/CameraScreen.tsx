import React, { useState, useRef, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ActivityIndicator, Alert } from 'react-native';
import { Camera } from 'expo-camera';
import { Audio } from 'expo-av';
import { useIsFocused } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import * as MediaLibrary from 'expo-media-library';

type CameraScreenProps = {
  onRecordingComplete: (uri: string) => void;
  onClose: () => void;
};

export default function CameraScreen({ onRecordingComplete, onClose }: CameraScreenProps) {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [cameraType, setCameraType] = useState(Camera.Constants.Type.back);
  const [isRecording, setIsRecording] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [saving, setSaving] = useState(false);
  const cameraRef = useRef<Camera | null>(null);
  const isFocused = useIsFocused();
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    (async () => {
      const { status: cameraStatus } = await Camera.requestCameraPermissionsAsync();
      const { status: audioStatus } = await Audio.requestPermissionsAsync();
      const { status: mediaLibraryStatus } = await MediaLibrary.requestPermissionsAsync();
      
      setHasPermission(
        cameraStatus === 'granted' && 
        audioStatus === 'granted' &&
        mediaLibraryStatus === 'granted'
      );
    })();

    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    };
  }, []);

  const startCountdown = () => {
    setCountdown(3);
    
    const countdownInterval = setInterval(() => {
      setCountdown((prev) => {
        if (prev === 1) {
          clearInterval(countdownInterval);
          startRecording();
          return null;
        }
        return prev ? prev - 1 : null;
      });
    }, 1000);
  };

  const startRecording = async () => {
    if (!cameraRef.current) return;
    
    try {
      setIsRecording(true);
      
      const { uri } = await cameraRef.current.recordAsync({
        maxDuration: 30, // Limit to 30 seconds
        quality: Camera.Constants.VideoQuality['720p'],
      });
      
      // Start the recording timer
      setRecordingTime(0);
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
      
      return uri;
    } catch (error) {
      console.error('Failed to start recording:', error);
      Alert.alert('Error', 'Failed to start video recording');
      setIsRecording(false);
    }
  };

  const stopRecording = async () => {
    if (!cameraRef.current || !isRecording) return;
    
    try {
      setSaving(true);
      setIsRecording(false);
      
      // Clear the recording timer
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
      
      // Stop recording
      cameraRef.current.stopRecording();
      
      // Wait for the video to be processed
      setTimeout(async () => {
        // At this point, the recordAsync promise should have resolved with the video URI
        // We'll handle this in the onRecordingComplete callback
        setSaving(false);
      }, 1000);
    } catch (error) {
      console.error('Failed to stop recording:', error);
      Alert.alert('Error', 'Failed to save video recording');
      setSaving(false);
      setIsRecording(false);
    }
  };

  const handleCameraReady = () => {
    console.log('Camera is ready');
  };

  const toggleCameraType = () => {
    setCameraType(
      cameraType === Camera.Constants.Type.back
        ? Camera.Constants.Type.front
        : Camera.Constants.Type.back
    );
  };

  // Format seconds into MM:SS
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  if (hasPermission === null) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#0077B6" />
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>
          To record tennis videos, TennisFlow needs access to your camera and microphone.
        </Text>
        <TouchableOpacity style={styles.permissionButton} onPress={onClose}>
          <Text style={styles.buttonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {isFocused && (
        <Camera
          ref={cameraRef}
          style={styles.camera}
          type={cameraType}
          onCameraReady={handleCameraReady}
          ratio="16:9"
        >
          <View style={styles.overlay}>
            {/* Top controls */}
            <View style={styles.topControls}>
              <TouchableOpacity style={styles.closeButton} onPress={onClose}>
                <Ionicons name="close" size={30} color="white" />
              </TouchableOpacity>
              <TouchableOpacity 
                style={styles.flipButton} 
                onPress={toggleCameraType}
                disabled={isRecording}
              >
                <Ionicons name="camera-reverse" size={30} color={isRecording ? "gray" : "white"} />
              </TouchableOpacity>
            </View>

            {/* Countdown overlay */}
            {countdown !== null && (
              <View style={styles.countdownOverlay}>
                <Text style={styles.countdownText}>{countdown}</Text>
              </View>
            )}

            {/* Recording time */}
            {isRecording && (
              <View style={styles.recordingInfo}>
                <View style={styles.recordingIndicator} />
                <Text style={styles.timeText}>{formatTime(recordingTime)}</Text>
              </View>
            )}

            {/* Bottom controls */}
            <View style={styles.bottomControls}>
              {!isRecording && !saving ? (
                <TouchableOpacity 
                  style={styles.recordButton} 
                  onPress={startCountdown}
                >
                  <View style={styles.recordButtonInner} />
                </TouchableOpacity>
              ) : (
                <TouchableOpacity 
                  style={styles.stopButton} 
                  onPress={stopRecording}
                  disabled={saving}
                >
                  {saving ? (
                    <ActivityIndicator color="white" size="small" />
                  ) : (
                    <View style={styles.stopButtonInner} />
                  )}
                </TouchableOpacity>
              )}
            </View>
          </View>
        </Camera>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
    justifyContent: 'center',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'transparent',
    justifyContent: 'space-between',
  },
  topControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 20,
    paddingTop: 40,
  },
  closeButton: {
    padding: 10,
  },
  flipButton: {
    padding: 10,
  },
  bottomControls: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingBottom: 40,
  },
  recordButton: {
    borderWidth: 3,
    borderColor: 'white',
    borderRadius: 35,
    height: 70,
    width: 70,
    alignItems: 'center',
    justifyContent: 'center',
  },
  recordButtonInner: {
    backgroundColor: '#FF3B30',
    borderRadius: 30,
    height: 60,
    width: 60,
  },
  stopButton: {
    borderWidth: 3,
    borderColor: 'white',
    borderRadius: 35,
    height: 70,
    width: 70,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stopButtonInner: {
    backgroundColor: 'white',
    height: 30,
    width: 30,
    borderRadius: 3,
  },
  countdownOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  countdownText: {
    fontSize: 80,
    fontWeight: 'bold',
    color: 'white',
  },
  recordingInfo: {
    position: 'absolute',
    top: 40,
    alignSelf: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 5,
  },
  recordingIndicator: {
    height: 10,
    width: 10,
    borderRadius: 5,
    backgroundColor: 'red',
    marginRight: 10,
  },
  timeText: {
    color: 'white',
    fontWeight: 'bold',
  },
  permissionText: {
    textAlign: 'center',
    padding: 20,
    color: 'white',
    fontSize: 16,
  },
  permissionButton: {
    backgroundColor: '#0077B6',
    padding: 15,
    borderRadius: 8,
    margin: 20,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
});