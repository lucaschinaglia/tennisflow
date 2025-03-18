import React, { useState, useRef, useEffect } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TouchableOpacity, 
  ActivityIndicator, 
  Alert,
  Dimensions,
  Image,
  ScrollView
} from 'react-native';
import { Video, AVPlaybackStatus } from 'expo-av';
import Slider from '@react-native-community/slider';
import * as VideoThumbnails from 'expo-video-thumbnails';
import * as VideoManipulator from 'expo-video-manipulator';
import * as FileSystem from 'expo-file-system';
import { Ionicons } from '@expo/vector-icons';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const THUMBNAIL_WIDTH = 60;

interface VideoEditorScreenProps {
  videoUri: string;
  onSave: (uri: string, thumbnailUri: string, duration: number) => void;
  onCancel: () => void;
}

export default function VideoEditorScreen({ videoUri, onSave, onCancel }: VideoEditorScreenProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [position, setPosition] = useState(0);
  const [thumbnails, setThumbnails] = useState<string[]>([]);
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [isTrimming, setIsTrimming] = useState(false);
  const [thumbnailUri, setThumbnailUri] = useState<string | null>(null);
  
  const videoRef = useRef<Video | null>(null);
  const positionUpdateInterval = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (positionUpdateInterval.current) {
        clearInterval(positionUpdateInterval.current);
      }
    };
  }, []);

  useEffect(() => {
    const loadVideo = async () => {
      try {
        setIsLoading(true);
        await generateThumbnails();
        setIsLoading(false);
      } catch (error) {
        console.error('Error loading video:', error);
        Alert.alert('Error', 'Failed to load video');
        onCancel();
      }
    };

    loadVideo();
  }, [videoUri]);

  const generateThumbnails = async () => {
    try {
      const { uri, width, height } = await VideoThumbnails.getThumbnailAsync(
        videoUri,
        {
          time: 0,
          quality: 0.5,
        }
      );
      
      setThumbnailUri(uri);
      
      // Get video information
      const status = await videoRef.current?.getStatusAsync();
      const videoDuration = status?.durationMillis ? status.durationMillis / 1000 : 0;
      
      setDuration(videoDuration);
      setTrimEnd(videoDuration);
      
      // Generate multiple thumbnails for the trimming slider
      const thumbnailCount = Math.ceil(SCREEN_WIDTH / THUMBNAIL_WIDTH);
      const newThumbnails = [];
      
      for (let i = 0; i < thumbnailCount; i++) {
        const time = (i / (thumbnailCount - 1)) * videoDuration;
        const thumbnail = await VideoThumbnails.getThumbnailAsync(
          videoUri,
          {
            time: time * 1000,
            quality: 0.2,
          }
        );
        newThumbnails.push(thumbnail.uri);
      }
      
      setThumbnails(newThumbnails);
    } catch (error) {
      console.error('Error generating thumbnails:', error);
      throw error;
    }
  };

  const handlePlaybackStatusUpdate = (status: AVPlaybackStatus) => {
    if (status.isLoaded) {
      setPosition(status.positionMillis / 1000);
      setDuration(status.durationMillis ? status.durationMillis / 1000 : 0);
      
      if (status.didJustFinish) {
        setIsPlaying(false);
      }
    }
  };

  const togglePlayPause = () => {
    if (isPlaying) {
      videoRef.current?.pauseAsync();
    } else {
      // If at the end, start from beginning
      if (position >= trimEnd) {
        videoRef.current?.setPositionAsync(trimStart * 1000);
      }
      videoRef.current?.playAsync();
    }
    setIsPlaying(!isPlaying);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleSliderChange = (value: number) => {
    setPosition(value);
    videoRef.current?.setPositionAsync(value * 1000);
  };

  const handleTrimStartChange = (value: number) => {
    // Ensure trim start is before trim end with at least 1 second buffer
    const newTrimStart = Math.min(value, trimEnd - 1);
    setTrimStart(newTrimStart);
    
    // If current position is before new trim start, update position
    if (position < newTrimStart) {
      setPosition(newTrimStart);
      videoRef.current?.setPositionAsync(newTrimStart * 1000);
    }
  };

  const handleTrimEndChange = (value: number) => {
    // Ensure trim end is after trim start with at least 1 second buffer
    const newTrimEnd = Math.max(value, trimStart + 1);
    setTrimEnd(newTrimEnd);
    
    // If current position is after new trim end, update position
    if (position > newTrimEnd) {
      setPosition(newTrimEnd);
      videoRef.current?.setPositionAsync(newTrimEnd * 1000);
    }
  };

  const saveVideo = async () => {
    try {
      setIsTrimming(true);
      
      // Create a thumbnail at the midpoint of the trimmed section
      const middleTime = (trimStart + trimEnd) / 2;
      const thumbnail = await VideoThumbnails.getThumbnailAsync(
        videoUri,
        {
          time: middleTime * 1000,
          quality: 0.7,
        }
      );
      
      // Trim video
      const startMillis = Math.floor(trimStart * 1000);
      const endMillis = Math.floor(trimEnd * 1000);
      const trimmedDuration = trimEnd - trimStart;
      
      // Simulate video trimming (since expo-video-manipulator is a mock package)
      // In a real app, use a library like react-native-video-processing or FFmpeg
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // For now, we'll just use the original video file as a placeholder
      // In a real implementation, you would process the video here
      const trimmedVideoUri = videoUri;
      
      // Save the trimmed video
      onSave(trimmedVideoUri, thumbnail.uri, trimmedDuration);
    } catch (error) {
      console.error('Error saving video:', error);
      Alert.alert('Error', 'Failed to save video');
    } finally {
      setIsTrimming(false);
    }
  };

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0077B6" />
        <Text style={styles.loadingText}>Loading video...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.videoContainer}>
        <Video
          ref={videoRef}
          source={{ uri: videoUri }}
          style={styles.video}
          resizeMode="contain"
          onPlaybackStatusUpdate={handlePlaybackStatusUpdate}
          shouldPlay={false}
          isLooping={false}
        />
        
        <TouchableOpacity 
          style={styles.playPauseButton} 
          onPress={togglePlayPause}
        >
          <Ionicons 
            name={isPlaying ? 'pause' : 'play'} 
            size={30} 
            color="white" 
          />
        </TouchableOpacity>
      </View>
      
      <View style={styles.controls}>
        <Text style={styles.timeText}>
          {formatTime(position)} / {formatTime(duration)}
        </Text>
        
        <Slider
          style={styles.slider}
          minimumValue={0}
          maximumValue={duration}
          value={position}
          onValueChange={handleSliderChange}
          minimumTrackTintColor="#0077B6"
          maximumTrackTintColor="#ddd"
          thumbTintColor="#0077B6"
        />
      </View>
      
      <View style={styles.trimSection}>
        <Text style={styles.sectionTitle}>Trim Video</Text>
        
        <View style={styles.trimControls}>
          <Text style={styles.trimTimeText}>Start: {formatTime(trimStart)}</Text>
          <Text style={styles.trimTimeText}>End: {formatTime(trimEnd)}</Text>
        </View>
        
        <View style={styles.thumbnailContainer}>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            {thumbnails.map((uri, index) => (
              <Image 
                key={index}
                source={{ uri }}
                style={styles.thumbnail}
              />
            ))}
          </ScrollView>
          
          <View style={styles.trimSliderContainer}>
            <Slider
              style={styles.trimSlider}
              minimumValue={0}
              maximumValue={duration}
              value={trimStart}
              onValueChange={handleTrimStartChange}
              minimumTrackTintColor="#ddd"
              maximumTrackTintColor="#0077B6"
              thumbTintColor="#0077B6"
            />
            
            <Slider
              style={styles.trimSlider}
              minimumValue={0}
              maximumValue={duration}
              value={trimEnd}
              onValueChange={handleTrimEndChange}
              minimumTrackTintColor="#0077B6"
              maximumTrackTintColor="#ddd"
              thumbTintColor="#0077B6"
            />
          </View>
        </View>
      </View>
      
      <View style={styles.actionButtons}>
        <TouchableOpacity 
          style={[styles.button, styles.cancelButton]} 
          onPress={onCancel}
          disabled={isTrimming}
        >
          <Text style={styles.buttonText}>Cancel</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.button, styles.saveButton]} 
          onPress={saveVideo}
          disabled={isTrimming}
        >
          {isTrimming ? (
            <ActivityIndicator size="small" color="white" />
          ) : (
            <Text style={styles.buttonText}>Save</Text>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  videoContainer: {
    height: 300,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  video: {
    width: '100%',
    height: '100%',
  },
  playPauseButton: {
    position: 'absolute',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    width: 60,
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  controls: {
    padding: 15,
  },
  slider: {
    width: '100%',
    height: 40,
  },
  timeText: {
    textAlign: 'center',
    fontSize: 16,
    color: '#333',
  },
  trimSection: {
    padding: 15,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  trimControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  trimTimeText: {
    fontSize: 14,
    color: '#666',
  },
  thumbnailContainer: {
    marginVertical: 10,
  },
  thumbnail: {
    width: THUMBNAIL_WIDTH,
    height: 40,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  trimSliderContainer: {
    marginTop: 10,
  },
  trimSlider: {
    width: '100%',
    height: 40,
  },
  actionButtons: {
    flexDirection: 'row',
    padding: 15,
    borderTopWidth: 1,
    borderTopColor: '#eee',
    justifyContent: 'space-between',
  },
  button: {
    paddingVertical: 12,
    paddingHorizontal: 30,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    flex: 1,
    marginHorizontal: 5,
  },
  cancelButton: {
    backgroundColor: '#f0f0f0',
  },
  saveButton: {
    backgroundColor: '#0077B6',
  },
  buttonText: {
    fontWeight: 'bold',
    fontSize: 16,
    color: '#fff',
  },
});