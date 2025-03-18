import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Dimensions,
  Image,
  Alert,
} from 'react-native';
import { Video, AVPlaybackStatus } from 'expo-av';
import Slider from '@react-native-community/slider';
import { Ionicons } from '@expo/vector-icons';
import * as AnalysisService from '../services/analysisService';
import * as VideoService from '../services/videoService';
import { SwingMetrics, FrameAnalysis } from '../services/analysisService';
import { useAuth } from '../hooks/useAuth';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

type VideoAnalysisScreenProps = {
  videoId: string;
  onClose?: () => void;
};

export default function VideoAnalysisScreen({ videoId, onClose }: VideoAnalysisScreenProps) {
  const [loading, setLoading] = useState(true);
  const [video, setVideo] = useState<any>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisService.VideoAnalysisResults | null>(null);
  const [currentFrame, setCurrentFrame] = useState<FrameAnalysis | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentSwing, setCurrentSwing] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'technique' | 'feedback'>('overview');
  const [frameDetails, setFrameDetails] = useState<FrameAnalysis[]>([]);
  const [analysisStatus, setAnalysisStatus] = useState<string>('pending');
  
  const videoRef = useRef<Video | null>(null);
  const { user } = useAuth();
  
  useEffect(() => {
    loadVideoAndAnalysis();
  }, [videoId]);
  
  const loadVideoAndAnalysis = async () => {
    try {
      setLoading(true);
      
      // Get the video details
      const videoDetails = await VideoService.getVideoDetails(videoId);
      setVideo(videoDetails);
      
      // Get the analysis status
      const status = await AnalysisService.getAnalysisStatus(videoId);
      setAnalysisStatus(status);
      
      if (status === 'completed') {
        // Get the analysis results
        const results = await AnalysisService.getVideoAnalysis(videoId);
        setAnalysisResults(results);
        
        if (results && results.frames.length > 0) {
          setCurrentFrame(results.frames[0]);
        }
        
        if (results && results.swings.length > 0) {
          setCurrentSwing(results.swings[0]);
        }
      }
      
      setLoading(false);
    } catch (error) {
      console.error('Error loading video and analysis:', error);
      Alert.alert('Error', 'Failed to load video analysis');
      if (onClose) onClose();
    }
  };
  
  const handlePlaybackStatusUpdate = (status: AVPlaybackStatus) => {
    if (status.isLoaded) {
      setPosition(status.positionMillis / 1000);
      setDuration(status.durationMillis ? status.durationMillis / 1000 : 0);
      
      if (analysisResults) {
        // Find the current frame based on position
        const currentTimeMs = status.positionMillis;
        const nearestFrame = findNearestFrame(currentTimeMs);
        if (nearestFrame) {
          setCurrentFrame(nearestFrame);
        }
        
        // Find the current swing
        const currentSwing = findCurrentSwing(currentTimeMs / 1000);
        if (currentSwing) {
          setCurrentSwing(currentSwing);
        }
      }
      
      if (status.didJustFinish) {
        setIsPlaying(false);
      }
    }
  };
  
  const findNearestFrame = (timeMs: number): FrameAnalysis | null => {
    if (!analysisResults || !analysisResults.frames.length) return null;
    
    // Find the frame closest to the current time
    return analysisResults.frames.reduce((prev, curr) => {
      const prevDiff = Math.abs(prev.timestamp * 1000 - timeMs);
      const currDiff = Math.abs(curr.timestamp * 1000 - timeMs);
      return currDiff < prevDiff ? curr : prev;
    });
  };
  
  const findCurrentSwing = (timeSec: number): any => {
    if (!analysisResults || !analysisResults.swings.length) return null;
    
    // Find a swing that contains the current time
    return analysisResults.swings.find(swing => 
      timeSec >= swing.startTime && timeSec <= swing.endTime
    ) || null;
  };
  
  const togglePlayPause = () => {
    if (isPlaying) {
      videoRef.current?.pauseAsync();
    } else {
      videoRef.current?.playAsync();
    }
    setIsPlaying(!isPlaying);
  };
  
  const handleSliderChange = (value: number) => {
    setPosition(value);
    videoRef.current?.setPositionAsync(value * 1000);
  };
  
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };
  
  const jumpToSwing = (swing: any) => {
    if (videoRef.current) {
      videoRef.current.setPositionAsync(swing.startTime * 1000);
      setPosition(swing.startTime);
      setCurrentSwing(swing);
    }
  };
  
  const jumpToKeyFrame = (frameIndex: number) => {
    if (!analysisResults || !videoRef.current) return;
    
    const frame = analysisResults.frames[frameIndex];
    videoRef.current.setPositionAsync(frame.timestamp * 1000);
    setPosition(frame.timestamp);
    setCurrentFrame(frame);
  };
  
  const renderMetricBar = (name: string, value: number, max: number) => {
    const percentage = (value / max) * 100;
    let color = '#4CAF50'; // Green
    
    if (percentage < 40) {
      color = '#F44336'; // Red
    } else if (percentage < 70) {
      color = '#FFC107'; // Yellow
    }
    
    return (
      <View style={styles.metricItem} key={name}>
        <View style={styles.metricNameContainer}>
          <Text style={styles.metricName}>{name}</Text>
          <Text style={styles.metricValue}>{value.toFixed(1)}</Text>
        </View>
        <View style={styles.metricBarContainer}>
          <View 
            style={[
              styles.metricBar, 
              { width: `${percentage}%`, backgroundColor: color }
            ]} 
          />
        </View>
      </View>
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0077B6" />
        <Text style={styles.loadingText}>Loading analysis...</Text>
      </View>
    );
  }

  if (analysisStatus !== 'completed') {
    return (
      <View style={styles.loadingContainer}>
        <Text style={styles.statusTitle}>Analysis Status: {analysisStatus}</Text>
        
        {analysisStatus === 'pending' && (
          <>
            <Text style={styles.statusText}>
              Your video is waiting to be analyzed. This process usually takes a few minutes.
            </Text>
            <ActivityIndicator size="large" color="#0077B6" style={{ marginTop: 20 }} />
          </>
        )}
        
        {analysisStatus === 'processing' && (
          <>
            <Text style={styles.statusText}>
              Your video is currently being analyzed. This process usually takes a few minutes.
            </Text>
            <ActivityIndicator size="large" color="#0077B6" style={{ marginTop: 20 }} />
          </>
        )}
        
        {analysisStatus === 'failed' && (
          <>
            <Text style={styles.statusText}>
              Unfortunately, the analysis of your video failed. This could be due to video quality or format issues.
            </Text>
            <TouchableOpacity 
              style={styles.retryButton}
              onPress={() => AnalysisService.requestAnalysis(videoId).then(() => loadVideoAndAnalysis())}
            >
              <Text style={styles.retryButtonText}>Retry Analysis</Text>
            </TouchableOpacity>
          </>
        )}
        
        <TouchableOpacity style={styles.closeButton} onPress={onClose}>
          <Text style={styles.closeButtonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Video Player */}
      <View style={styles.videoContainer}>
        <Video
          ref={videoRef}
          source={{ uri: video.video_url }}
          style={styles.video}
          resizeMode="contain"
          onPlaybackStatusUpdate={handlePlaybackStatusUpdate}
          shouldPlay={false}
          isLooping={false}
        />
        
        {currentFrame && currentFrame.poseData && (
          <View style={styles.overlayContainer}>
            {/* This would render skeleton overlay in a real app */}
            {currentFrame.annotations && currentFrame.annotations.map((annotation, index) => (
              <View 
                key={index}
                style={[
                  styles.annotation,
                  {
                    left: `${annotation.position.x * 100}%`,
                    top: `${annotation.position.y * 100}%`
                  }
                ]}
              >
                <Text style={styles.annotationText}>{annotation.text}</Text>
              </View>
            ))}
          </View>
        )}
        
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
        
        {currentFrame && (
          <View style={styles.phaseIndicator}>
            <Text style={styles.phaseText}>
              {currentFrame.swingPhase ? currentFrame.swingPhase.charAt(0).toUpperCase() + currentFrame.swingPhase.slice(1) : 'No Swing'}
            </Text>
          </View>
        )}
      </View>
      
      {/* Video Controls */}
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
      
      {/* Tab Navigation */}
      <View style={styles.tabContainer}>
        <TouchableOpacity 
          style={[styles.tab, activeTab === 'overview' && styles.activeTab]} 
          onPress={() => setActiveTab('overview')}
        >
          <Text style={[styles.tabText, activeTab === 'overview' && styles.activeTabText]}>Overview</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.tab, activeTab === 'technique' && styles.activeTab]} 
          onPress={() => setActiveTab('technique')}
        >
          <Text style={[styles.tabText, activeTab === 'technique' && styles.activeTabText]}>Technique</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.tab, activeTab === 'feedback' && styles.activeTab]} 
          onPress={() => setActiveTab('feedback')}
        >
          <Text style={[styles.tabText, activeTab === 'feedback' && styles.activeTabText]}>Feedback</Text>
        </TouchableOpacity>
      </View>
      
      {/* Content Area */}
      <ScrollView style={styles.contentContainer}>
        {/* Overview Tab Content */}
        {activeTab === 'overview' && analysisResults && (
          <View style={styles.overviewContainer}>
            <Text style={styles.sectionTitle}>Swing Summary</Text>
            
            <View style={styles.summaryCard}>
              <View style={styles.summaryRow}>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>Swing Type</Text>
                  <Text style={styles.summaryValue}>{analysisResults.summary.swingType}</Text>
                </View>
                
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>Swing Count</Text>
                  <Text style={styles.summaryValue}>{analysisResults.summary.swingCount}</Text>
                </View>
              </View>
              
              <View style={styles.divider} />
              
              <Text style={styles.metricSectionTitle}>Performance Metrics</Text>
              
              {Object.entries(analysisResults.summary.averageMetrics).map(([key, value]) => 
                renderMetricBar(key.replace(/([A-Z])/g, ' $1').trim(), value, 100)
              )}
            </View>
            
            <Text style={styles.sectionTitle}>Strengths</Text>
            <View style={styles.listCard}>
              {analysisResults.summary.strengths.map((strength, index) => (
                <View key={index} style={styles.listItem}>
                  <Ionicons name="checkmark-circle" size={20} color="#4CAF50" />
                  <Text style={styles.listText}>{strength}</Text>
                </View>
              ))}
            </View>
            
            <Text style={styles.sectionTitle}>Areas to Improve</Text>
            <View style={styles.listCard}>
              {analysisResults.summary.weaknesses.map((weakness, index) => (
                <View key={index} style={styles.listItem}>
                  <Ionicons name="alert-circle" size={20} color="#F44336" />
                  <Text style={styles.listText}>{weakness}</Text>
                </View>
              ))}
            </View>
          </View>
        )}
        
        {/* Technique Tab Content */}
        {activeTab === 'technique' && analysisResults && (
          <View style={styles.techniqueContainer}>
            <Text style={styles.sectionTitle}>Swings</Text>
            
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.swingsScroll}>
              {analysisResults.swings.map((swing, index) => (
                <TouchableOpacity 
                  key={index} 
                  style={[
                    styles.swingCard,
                    currentSwing?.id === swing.id && styles.activeSwingCard
                  ]}
                  onPress={() => jumpToSwing(swing)}
                >
                  <Text style={styles.swingType}>{swing.swingType}</Text>
                  <Text style={styles.swingTime}>{formatTime(swing.startTime)}</Text>
                  <View style={styles.swingScore}>
                    <Text style={styles.swingScoreText}>{swing.score}</Text>
                  </View>
                </TouchableOpacity>
              ))}
            </ScrollView>
            
            {currentSwing && (
              <>
                <Text style={styles.sectionTitle}>Swing Details</Text>
                
                <View style={styles.detailCard}>
                  <View style={styles.detailHeader}>
                    <Text style={styles.detailTitle}>{currentSwing.swingType} Swing</Text>
                    <Text style={styles.detailScore}>Score: {currentSwing.score}/100</Text>
                  </View>
                  
                  <View style={styles.divider} />
                  
                  <Text style={styles.metricSectionTitle}>Metrics</Text>
                  
                  {Object.entries(currentSwing.metrics).map(([key, value]) => 
                    renderMetricBar(key.replace(/([A-Z])/g, ' $1').trim(), value as number, 100)
                  )}
                  
                  <Text style={styles.keyFramesTitle}>Key Frames</Text>
                  
                  <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.keyFramesScroll}>
                    {currentSwing.keyFrames.map((frameIndex: number, index: number) => {
                      const frame = analysisResults.frames[frameIndex];
                      return (
                        <TouchableOpacity 
                          key={index} 
                          style={styles.keyFrameCard}
                          onPress={() => jumpToKeyFrame(frameIndex)}
                        >
                          <Text style={styles.keyFramePhase}>{frame.swingPhase || 'Unknown'}</Text>
                          <Text style={styles.keyFrameTime}>{formatTime(frame.timestamp)}</Text>
                        </TouchableOpacity>
                      );
                    })}
                  </ScrollView>
                </View>
              </>
            )}
          </View>
        )}
        
        {/* Feedback Tab Content */}
        {activeTab === 'feedback' && analysisResults && (
          <View style={styles.feedbackContainer}>
            <Text style={styles.sectionTitle}>Coach Recommendations</Text>
            
            <View style={styles.listCard}>
              {analysisResults.summary.improvementSuggestions.map((suggestion, index) => (
                <View key={index} style={styles.feedbackItem}>
                  <View style={styles.feedbackHeader}>
                    <Ionicons name="bulb" size={20} color="#FFC107" />
                    <Text style={styles.feedbackTitle}>Improvement Tip</Text>
                  </View>
                  <Text style={styles.feedbackText}>{suggestion}</Text>
                </View>
              ))}
            </View>
            
            <Text style={styles.sectionTitle}>Training Drills</Text>
            
            <View style={styles.listCard}>
              <View style={styles.feedbackItem}>
                <View style={styles.feedbackHeader}>
                  <Ionicons name="fitness" size={20} color="#4CAF50" />
                  <Text style={styles.feedbackTitle}>Follow-Through Extension</Text>
                </View>
                <Text style={styles.feedbackText}>
                  Practice extending your follow-through by hitting against a wall and holding your finish position for 3 seconds after each shot.
                </Text>
              </View>
              
              <View style={styles.feedbackItem}>
                <View style={styles.feedbackHeader}>
                  <Ionicons name="fitness" size={20} color="#4CAF50" />
                  <Text style={styles.feedbackTitle}>Hip Rotation Drill</Text>
                </View>
                <Text style={styles.feedbackText}>
                  Shadow swings with a focus on hip rotation. Place a resistance band around your knees to increase awareness of hip movement.
                </Text>
              </View>
            </View>
            
            {/* User Notes Section - In a real app, this would allow users to add notes */}
            <Text style={styles.sectionTitle}>Your Notes</Text>
            
            <TouchableOpacity style={styles.addNoteButton}>
              <Ionicons name="add-circle" size={20} color="#0077B6" />
              <Text style={styles.addNoteText}>Add Note</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
      
      {/* Close Button */}
      {onClose && (
        <TouchableOpacity style={styles.closeButton} onPress={onClose}>
          <Text style={styles.closeButtonText}>Close</Text>
        </TouchableOpacity>
      )}
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
    padding: 20,
    backgroundColor: '#fff',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  statusTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  statusText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 20,
  },
  videoContainer: {
    height: 300,
    backgroundColor: '#000',
    position: 'relative',
  },
  video: {
    width: '100%',
    height: '100%',
  },
  overlayContainer: {
    ...StyleSheet.absoluteFillObject,
    // In a real app, this would contain SVG or Canvas elements for skeleton overlay
  },
  annotation: {
    position: 'absolute',
    backgroundColor: 'rgba(255, 255, 255, 0.7)',
    padding: 5,
    borderRadius: 5,
    transform: [{ translateX: -50 }, { translateY: -50 }],
  },
  annotationText: {
    fontSize: 12,
    color: '#333',
  },
  playPauseButton: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: [{ translateX: -25 }, { translateY: -25 }],
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
  },
  phaseIndicator: {
    position: 'absolute',
    bottom: 10,
    right: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 5,
  },
  phaseText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  controls: {
    padding: 10,
    backgroundColor: '#f8f8f8',
  },
  timeText: {
    textAlign: 'center',
    fontSize: 14,
    color: '#666',
  },
  slider: {
    width: '100%',
    height: 40,
  },
  tabContainer: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  tab: {
    flex: 1,
    paddingVertical: 15,
    alignItems: 'center',
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: '#0077B6',
  },
  tabText: {
    fontSize: 16,
    color: '#666',
  },
  activeTabText: {
    color: '#0077B6',
    fontWeight: 'bold',
  },
  contentContainer: {
    flex: 1,
    padding: 15,
  },
  overviewContainer: {},
  techniqueContainer: {},
  feedbackContainer: {},
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 10,
    marginBottom: 10,
    color: '#333',
  },
  summaryCard: {
    backgroundColor: '#f8f8f8',
    borderRadius: 8,
    padding: 15,
    marginBottom: 15,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  summaryItem: {
    flex: 1,
  },
  summaryLabel: {
    color: '#666',
    fontSize: 14,
  },
  summaryValue: {
    color: '#333',
    fontSize: 16,
    fontWeight: 'bold',
    textTransform: 'capitalize',
    marginTop: 5,
  },
  divider: {
    height: 1,
    backgroundColor: '#ddd',
    marginVertical: 10,
  },
  metricSectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  metricItem: {
    marginBottom: 8,
  },
  metricNameContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  metricName: {
    fontSize: 14,
    color: '#666',
  },
  metricValue: {
    fontSize: 14,
    color: '#333',
    fontWeight: 'bold',
  },
  metricBarContainer: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  metricBar: {
    height: '100%',
    borderRadius: 4,
  },
  listCard: {
    backgroundColor: '#f8f8f8',
    borderRadius: 8,
    padding: 15,
    marginBottom: 15,
  },
  listItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  listText: {
    fontSize: 14,
    color: '#333',
    marginLeft: 10,
    flex: 1,
  },
  swingsScroll: {
    marginBottom: 15,
  },
  swingCard: {
    width: 120,
    height: 80,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    padding: 10,
    marginRight: 10,
    justifyContent: 'space-between',
  },
  activeSwingCard: {
    backgroundColor: '#e6f7ff',
    borderWidth: 1,
    borderColor: '#0077B6',
  },
  swingType: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    textTransform: 'capitalize',
  },
  swingTime: {
    fontSize: 12,
    color: '#666',
  },
  swingScore: {
    position: 'absolute',
    top: 5,
    right: 5,
    backgroundColor: '#0077B6',
    width: 26,
    height: 26,
    borderRadius: 13,
    justifyContent: 'center',
    alignItems: 'center',
  },
  swingScoreText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  detailCard: {
    backgroundColor: '#f8f8f8',
    borderRadius: 8,
    padding: 15,
    marginBottom: 15,
  },
  detailHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  detailTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    textTransform: 'capitalize',
  },
  detailScore: {
    fontSize: 14,
    color: '#0077B6',
    fontWeight: 'bold',
  },
  keyFramesTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginTop: 15,
    marginBottom: 10,
    color: '#333',
  },
  keyFramesScroll: {
    marginBottom: 5,
  },
  keyFrameCard: {
    width: 100,
    height: 70,
    backgroundColor: '#e0e0e0',
    borderRadius: 8,
    padding: 8,
    marginRight: 10,
    justifyContent: 'space-between',
  },
  keyFramePhase: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#333',
    textTransform: 'capitalize',
  },
  keyFrameTime: {
    fontSize: 12,
    color: '#666',
  },
  feedbackItem: {
    marginBottom: 15,
  },
  feedbackHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
  },
  feedbackTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginLeft: 5,
  },
  feedbackText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  addNoteButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    padding: 15,
    marginBottom: 15,
  },
  addNoteText: {
    color: '#0077B6',
    fontWeight: 'bold',
    fontSize: 16,
    marginLeft: 5,
  },
  closeButton: {
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
    margin: 15,
  },
  closeButtonText: {
    color: '#333',
    fontWeight: 'bold',
    fontSize: 16,
  },
  retryButton: {
    backgroundColor: '#0077B6',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
    margin: 15,
  },
  retryButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
});