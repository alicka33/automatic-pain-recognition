import { Platform } from 'react-native';
import { API_UPLOAD_URL, HUGGING_FACE_TOKEN } from '@/constants/api';

const getMimeType = (ext: string): string => {
  switch (ext.toLowerCase()) {
    case 'mov': return 'video/quicktime';
    case 'mp4': return 'video/mp4';
    case 'avi': return 'video/x-msvideo';
    case 'webm': return 'video/webm';
    default: return 'video/mp4';
  }
};

export const uploadAndAnalyzeVideo = async (uri: string): Promise<string> => {
  const ext = uri.split('.').pop() || 'mp4';
  const mimeType = getMimeType(ext);
  const fileName = `video_${Date.now()}.${ext}`;
  const fileUri = Platform.OS === 'android' ? uri : uri.replace('file://', '');

  const formData = new FormData();
  formData.append('video', { uri: fileUri, name: fileName, type: mimeType } as any);

  const response = await fetch(API_UPLOAD_URL, {
    method: 'POST',
    headers: { Authorization: `Bearer ${HUGGING_FACE_TOKEN}` },
    body: formData,
  });

  if (!response.ok) throw new Error(`Server error: ${response.status}`);
  const data = await response.json();
  return data.painLevel;
};
