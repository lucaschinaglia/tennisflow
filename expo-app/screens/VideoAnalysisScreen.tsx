import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  StyleSheet,
  Text,
  SafeAreaView,
  ActivityIndicator,
  ScrollView,
  TouchableOpacity,
  Dimensions,
  Image,
} from 'react-native';
import { Video } from 'expo-av';
import Slider from '@react-native-community/slider';
import { Ionicons } from '@expo/vector-icons';
import { supabase } from '../services/supabaseService';
import { 
  getVideoAnalysis, 
  checkAnalysisStatus,
  AnalysisResult,
  TaskStatus,
  FrameData
} from '../services/analysisService';
import PoseVisualizer from '../components/PoseVisualizer';
import SwingMetrics from '../components/SwingMetrics';
import { StatusBar } from 'expo-status-bar';

interface VideoAnalysisScreenProps {
  route: {
    params: {
      videoId: string;
      taskId: string;
      videoUri?: string;
    };
  };
  navigation: any;
}

const POLLING_INTERVAL = 3000; // Poll status every 3 seconds

const VideoAnalysisScreen: React.FC<VideoAnalysisScreenProps> = ({ route, navigation }) => {
  const { videoId, taskId, videoUri } = route.params;
  
  const [loading, setLoading] = useState(true);
  const [processingStatus, setProcessingStatus] = useState<TaskStatus | null>(null);
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(videoUri || null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showPose, setShowPose] = useState(true);
  const [showMetrics, setShowMetrics] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const videoRef = useRef<Video>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  
  // Component dimensions
  const windowWidth = Dimensions.get('window').width;
  const videoHeight = windowWidth * 0.5625; // 16:9 aspect ratio
  
  // Get video URL from Supabase
  const getVideoUrl = async () => {
    if (videoUri) return; // Already have the URI
    
    try {
      const { data, error } = await supabase.storage
        .from('tennis-videos')
        .createSignedUrl(videoId, 3600); // 1 hour expiry
      
      if (error) throw error;
      if (data) setVideoUrl(data.signedUrl);
    } catch (err: any) {
      console.error('Error getting video URL:', err);
      setError(`Error loading video: ${err.message}`);
    }
  };
  
  // Poll for analysis status
  const startPolling = () => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    
    pollingRef.current = setInterval(async () => {
      try {
        const status = await checkAnalysisStatus(taskId);
        setProcessingStatus(status);
        
        if (status.status === 'completed') {
          stopPolling();
          fetchAnalysisData();
        } else if (status.status === 'failed') {
          stopPolling();
          setError(`Analysis failed: ${status.error || 'Unknown error'}`);
        }
      } catch (err: any) {
        console.error('Error checking status:', err);
        setError(`Error checking analysis status: ${err.message}`);
        stopPolling();
      }
    }, POLLING_INTERVAL);
  };
  
  const stopPolling = () => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  };
  
  // Fetch analysis data
  const fetchAnalysisData = async () => {
    try {
      const data = await getVideoAnalysis(videoId);
      setAnalysisData(data);
      setLoading(false);
    } catch (err: any) {
      console.error('Error fetching analysis:', err);
      setError(`Error fetching analysis: ${err.message}`);
      setLoading(false);
    }
  };
  
  // Initialize
  useEffect(() => {
    getVideoUrl();
    
    // Check initial status
    const checkInitialStatus = async () => {
      try {
        const status = await checkAnalysisStatus(taskId);
        setProcessingStatus(status);
        
        if (status.status === 'completed') {
          fetchAnalysisData();
        } else if (status.status === 'failed') {
          setError(`Analysis failed: ${status.error || 'Unknown error'}`);
          setLoading(false);
        } else {
          // Still processing, start polling
          startPolling();
        }
      } catch (err: any) {
        console.error('Error checking initial status:', err);
        setError(`Error checking analysis status: ${err.message}`);
        setLoading(false);
      }
    };
    
    checkInitialStatus();
    
    return () => {
      stopPolling();
    };
  }, []);
  
  // Handle video playback position change
  const onPositionChange = (position: number) => {
    if (!analysisData) return;
    
    // Find the closest frame to the current position
    const currentTimeInSeconds = position / 1000; // Convert ms to seconds
    let closestFrameIndex = 0;
    let minDiff = Number.MAX_VALUE;
    
    analysisData.frames.forEach((frame, index) => {
      const diff = Math.abs(frame.timestamp - currentTimeInSeconds);
      if (diff < minDiff) {
        minDiff = diff;
        closestFrameIndex = index;
      }
    });
    
    setCurrentFrame(closestFrameIndex);
  };
  
  // Get current frame data
  const getCurrentFrameData = (): FrameData | null => {
    if (!analysisData || analysisData.frames.length === 0) return null;
    return analysisData.frames[currentFrame];
  };
  
  const frameData = getCurrentFrameData();
  
  // Render loading state
  if (loading || !analysisData) {
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
          <Text style={styles.headerTitle}>Analysis in Progress</Text>
          <View style={styles.headerRight} />
        </View>
        
        <View style={styles.loadingContainer}>
          {error ? (
            <View style={styles.errorContainer}>
              <Ionicons name="alert-circle" size={48} color="#F44336" />
              <Text style={styles.errorText}>{error}</Text>
              <TouchableOpacity
                style={styles.retryButton}
                onPress={() => navigation.goBack()}
              >
                <Text style={styles.retryButtonText}>Back to Upload</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <>
              <ActivityIndicator size="large" color="#2196F3" />
              <Text style={styles.loadingText}>
                {processingStatus?.status === 'queued' && 'Video queued for analysis...'}
                {processingStatus?.status === 'processing' && `Processing video${processingStatus.progress ? ` (${Math.round(processingStatus.progress)}%)` : '...'}` }
              </Text>
              <Text style={styles.subLoadingText}>
                This may take a few minutes
              </Text>
            </>
          )}
        </View>
      </SafeAreaView>
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
        <Text style={styles.headerTitle}>Swing Analysis</Text>
        <View style={styles.headerRight} />
      </View>
      
      <ScrollView style={styles.scrollContainer}>
        {/* Video player */}
        {videoUrl && (
          <View style={styles.videoContainer}>
            <Video
              ref={videoRef}
              source={{ uri: videoUrl }}
              style={[styles.video, { height: videoHeight }]}
              useNativeControls={false}
              resizeMode="contain"
              isLooping
              shouldPlay={isPlaying}
              onPlaybackStatusUpdate={(status) => {
                if (status.isLoaded) {
                  onPositionChange(status.positionMillis);
                  
                  // Auto-pause at the end
                  if (status.didJustFinish) {
                    setIsPlaying(false);
                  }
                }
              }}
            />
            
            {/* Pose visualization overlay */}
            {showPose && frameData && (
              <View style={[styles.poseOverlay, { height: videoHeight }]}>
                <PoseVisualizer
                  poseData={frameData.poseData}
                  annotations={frameData.annotations}
                  swingPhase={frameData.swingPhase}
                  width={windowWidth}
                  height={videoHeight}
                />
              </View>
            )}
            
            {/* Player controls */}
            <View style={styles.controls}>
              <TouchableOpacity
                style={styles.controlButton}
                onPress={() => setIsPlaying(!isPlaying)}
              >
                <Ionicons
                  name={isPlaying ? 'pause' : 'play'}
                  size={24}
                  color="#FFF"
                />
              </TouchableOpacity>
              
              <Slider
                style={styles.slider}
                minimumValue={0}
                maximumValue={analysisData.duration * 1000} // Convert to milliseconds
                value={frameData ? frameData.timestamp * 1000 : 0}
                onValueChange={(value) => {
                  if (videoRef.current) {
                    videoRef.current.setPositionAsync(value);
                  }
                }}
                minimumTrackTintColor="#2196F3"
                maximumTrackTintColor="#FFFFFF33"
                thumbTintColor="#2196F3"
              />
              
              <TouchableOpacity
                style={styles.controlButton}
                onPress={() => setShowPose(!showPose)}
              >
                <Ionicons
                  name={showPose ? 'eye' : 'eye-off'}
                  size={24}
                  color="#FFF"
                />
              </TouchableOpacity>
            </View>
          </View>
        )}
        
        {/* Analysis tabs */}
        <View style={styles.tabsContainer}>
          <TouchableOpacity
            style={[
              styles.tab,
              !showMetrics && styles.activeTab
            ]}
            onPress={() => setShowMetrics(false)}
          >
            <Text style={[
              styles.tabText,
              !showMetrics && styles.activeTabText
            ]}>Analysis</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.tab,
              showMetrics && styles.activeTab
            ]}
            onPress={() => setShowMetrics(true)}
          >
            <Text style={[
              styles.tabText,
              showMetrics && styles.activeTabText
            ]}>Metrics</Text>
          </TouchableOpacity>
        </View>
        
        {/* Analysis content */}
        {!showMetrics ? (
          // Frame by frame analysis
          <View style={styles.frameAnalysisContainer}>
            {/* Phase indicator */}
            {frameData && (
              <View style={styles.phaseContainer}>
                <Text style={styles.phaseLabel}>Phase:</Text>
                <View style={styles.phaseProgressContainer}>
                  {['preparation', 'backswing', 'forward-swing', 'contact', 'follow-through'].map(phase => (
                    <View 
                      key={phase}
                      style={[
                        styles.phaseItem,
                        frameData.swingPhase === phase && styles.activePhaseItem
                      ]}
                    >
                      <Text 
                        style={[
                          styles.phaseItemText,
                          frameData.swingPhase === phase && styles.activePhaseItemText
                        ]}
                      >
                        {phase.replace('-', ' ')}
                      </Text>
                    </View>
                  ))}
                </View>
              </View>
            )}
            
            {/* Current frame annotations */}
            {frameData && frameData.annotations.map((annotation, index) => (
              <View key={`annotation-${index}`} style={styles.annotationItem}>
                <View 
                  style={[
                    styles.annotationBullet,
                    { backgroundColor: annotation.color || '#2196F3' }
                  ]} 
                />
                <Text style={styles.annotationText}>{annotation.text}</Text>
              </View>
            ))}
            
            {/* Key frame navigation */}
            {analysisData.swings[0]?.keyFrames && (
              <View style={styles.keyFramesSection}>
                <Text style={styles.sectionTitle}>Key Moments</Text>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.keyFramesContainer}>
                  {analysisData.swings[0].keyFrames.map((frameIndex, index) => {
                    const phase = analysisData.frames[frameIndex]?.swingPhase || '';
                    return (
                      <TouchableOpacity 
                        key={`keyframe-${index}`}
                        style={styles.keyFrameItem}
                        onPress={() => {
                          setCurrentFrame(frameIndex);
                          if (videoRef.current) {
                            const timeInMs = analysisData.frames[frameIndex].timestamp * 1000;
                            videoRef.current.setPositionAsync(timeInMs);
                          }
                        }}
                      >
                        <View style={styles.keyFrameImageContainer}>
                          {/* This would ideally be an extracted frame from the video */}
                          <View style={styles.keyFrameImagePlaceholder}>
                            <Text style={styles.keyFrameNumber}>{index + 1}</Text>
                          </View>
                        </View>
                        <Text style={styles.keyFramePhase}>{phase.replace('-', ' ')}</Text>
                      </TouchableOpacity>
                    );
                  })}
                </ScrollView>
              </View>
            )}
          </View>
        ) : (
          // Metrics view
          <SwingMetrics
            metrics={analysisData.summary.averageMetrics}
            strengths={analysisData.summary.strengths}
            weaknesses={analysisData.summary.weaknesses}
            suggestions={analysisData.summary.improvementSuggestions}
            swingType={analysisData.summary.swingType}
            score={analysisData.swings[0]?.score}
          />
        )}
      </ScrollView>
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
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#111',
  },
  headerTitle: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  backButton: {
    padding: 8,
  },
  headerRight: {
    width: 40,
  },
  scrollContainer: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  videoContainer: {
    position: 'relative',
    backgroundColor: '#000',
  },
  video: {
    width: '100%',
  },
  poseOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'transparent',
  },
  controls: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  controlButton: {
    padding: 8,
  },
  slider: {
    flex: 1,
    marginHorizontal: 8,
  },
  tabsContainer: {
    flexDirection: 'row',
    backgroundColor: '#FFF',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 1,
  },
  tab: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: '#2196F3',
  },
  tabText: {
    fontSize: 16,
    color: '#666',
  },
  activeTabText: {
    color: '#2196F3',
    fontWeight: 'bold',
  },
  frameAnalysisContainer: {
    padding: 16,
    backgroundColor: '#FFF',
  },
  phaseContainer: {
    marginBottom: 16,
  },
  phaseLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#333',
  },
  phaseProgressContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  phaseItem: {
    paddingVertical: 8,
    paddingHorizontal: 4,
    borderRadius: 4,
    flex: 1,
    marginHorizontal: 2,
    backgroundColor: '#F1F1F1',
    alignItems: 'center',
  },
  activePhaseItem: {
    backgroundColor: '#2196F3',
  },
  phaseItemText: {
    fontSize: 10,
    color: '#666',
    textAlign: 'center',
  },
  activePhaseItemText: {
    color: '#FFF',
    fontWeight: 'bold',
  },
  annotationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    padding: 12,
    backgroundColor: '#F9F9F9',
    borderRadius: 8,
  },
  annotationBullet: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  annotationText: {
    flex: 1,
    fontSize: 14,
    color: '#333',
  },
  keyFramesSection: {
    marginTop: 24,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#333',
  },
  keyFramesContainer: {
    flexDirection: 'row',
    marginBottom: 16,
  },
  keyFrameItem: {
    marginRight: 12,
    alignItems: 'center',
    width: 80,
  },
  keyFrameImageContainer: {
    width: 80,
    height: 60,
    backgroundColor: '#EEE',
    borderRadius: 8,
    overflow: 'hidden',
    marginBottom: 4,
  },
  keyFrameImagePlaceholder: {
    width: '100%',
    height: '100%',
    backgroundColor: '#DDD',
    justifyContent: 'center',
    alignItems: 'center',
  },
  keyFrameNumber: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#888',
  },
  keyFramePhase: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 20,
    fontSize: 16,
    color: '#FFF',
    textAlign: 'center',
  },
  subLoadingText: {
    marginTop: 8,
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
  },
  errorContainer: {
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    marginTop: 16,
    fontSize: 16,
    color: '#F44336',
    textAlign: 'center',
  },
  retryButton: {
    marginTop: 20,
    backgroundColor: '#2196F3',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 25,
  },
  retryButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default VideoAnalysisScreen;