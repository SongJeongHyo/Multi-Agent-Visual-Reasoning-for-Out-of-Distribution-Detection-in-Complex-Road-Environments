#!/usr/bin/env python3
"""
dataset.py - Dataset classes for anomaly detection datasets
각 데이터셋의 특성에 맞는 전용 클래스들
"""

import os
import numpy as np
from PIL import Image
from glob import glob
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class BaseAnomalyDataset(Dataset, ABC):
    """기본 이상 탐지 데이터셋 클래스"""
    
    def __init__(self, dataset_dir, split='all'):
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_paths = []
        self.label_paths = []
        self.dataset_name = self.__class__.__name__
        
        # 데이터 로딩
        self._load_data()
        
        print(f"Loaded {self.dataset_name}: {len(self.image_paths)} samples")
        if len(self.image_paths) > 0:
            print(f"  Sample files: {os.path.basename(self.image_paths[0])} -> {os.path.basename(self.label_paths[0])}")
    
    @abstractmethod
    def _load_data(self):
        """데이터셋별 데이터 로딩 구현"""
        pass
    
    @abstractmethod
    def _process_mask(self, mask):
        """데이터셋별 마스크 처리 구현"""
        pass
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # 이미지 로딩
        image = Image.open(image_path).convert("RGB")
        
        # 마스크 로딩 및 처리
        mask = np.array(Image.open(label_path))
        binary_mask = self._process_mask(mask)
        
        return {
            'image': image,
            'mask': binary_mask,
            'image_path': image_path,
            'label_path': label_path,
            'image_name': os.path.basename(image_path)
        }
    
    def get_sample_info(self, idx):
        """샘플 정보 반환"""
        sample = self[idx]
        return {
            'index': idx,
            'image_name': sample['image_name'],
            'image_size': sample['image'].size,
            'mask_shape': sample['mask'].shape,
            'positive_pixels': int(np.sum(sample['mask'])),
            'positive_ratio': float(np.sum(sample['mask']) / sample['mask'].size)
        }


class RoadAnomalyDataset(BaseAnomalyDataset):
    """Road Anomaly 데이터셋
    - 마스크 값: 0 (배경), 1 (이상)
    - 파일 형식: .jpg -> .png
    """
    
    def _load_data(self):
        original_dir = os.path.join(self.dataset_dir, 'original')
        label_dir = os.path.join(self.dataset_dir, 'labels')
        
        if not os.path.exists(original_dir) or not os.path.exists(label_dir):
            print(f"Warning: Directory not found - {original_dir} or {label_dir}")
            return
        
        # 이미지 파일 찾기
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(original_dir, ext)))
        
        # 매칭되는 라벨 찾기
        for image_path in sorted(image_files):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(label_dir, f"{base_name}.png")
            
            if os.path.exists(label_path):
                self.image_paths.append(image_path)
                self.label_paths.append(label_path)
            else:
                print(f"Warning: No label found for {base_name}")
    
    def _process_mask(self, mask):
        """Road Anomaly: 0이 아닌 모든 값을 positive로 처리"""
        # 마스크 차원 확인
        if len(mask.shape) == 3:
            # RGB 마스크인 경우 - 평균내서 그레이스케일로
            mask = mask.mean(axis=2)
        
        # 0이 아닌 값을 positive로 처리 (0-1 범위 마스크)
        binary_mask = (mask > 0).astype(bool)
        
        return binary_mask


class FishyscapesDataset(BaseAnomalyDataset):
    """Fishyscapes 데이터셋 (LostAndFound, Static)
    - 마스크 값: 0 (배경), 1 (OOD 객체), 255 (알려진 객체)
    - OOD 탐지이므로 값이 1인 픽셀만 positive로 처리
    - 파일 형식: .png -> .png
    """
    
    def _load_data(self):
        original_dir = os.path.join(self.dataset_dir, 'original')
        label_dir = os.path.join(self.dataset_dir, 'labels')
        
        if not os.path.exists(original_dir) or not os.path.exists(label_dir):
            print(f"Warning: Directory not found - {original_dir} or {label_dir}")
            return
        
        # PNG 이미지 파일 찾기
        image_files = glob(os.path.join(original_dir, '*.png'))
        
        # 매칭되는 라벨 찾기
        for image_path in sorted(image_files):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(label_dir, f"{base_name}.png")
            
            if os.path.exists(label_path):
                self.image_paths.append(image_path)
                self.label_paths.append(label_path)
            else:
                print(f"Warning: No label found for {base_name}")
    
    def _process_mask(self, mask):
        """Fishyscapes: 값이 1인 픽셀만 positive로 처리 (OOD 객체)"""
        # Fishyscapes 마스크 값 의미:
        # 0: 배경 (정상)
        # 1: OOD 객체 (이상) <- 이것만 positive로 처리
        # 255: 알려진 객체 (정상)
        
        if len(mask.shape) == 3:
            # RGB 마스크인 경우 첫 번째 채널 사용
            mask = mask[:, :, 0]
        
        # 오직 값이 1인 픽셀만 positive로 처리
        binary_mask = (mask == 1).astype(bool)
        
        return binary_mask


class SegmentMeDataset(BaseAnomalyDataset):
    """Segment Me 데이터셋 (AnomalyTrack, ObstacleTrack)
    - validation 시리즈만 라벨이 있음
    - 라벨 파일명: validation*_labels_semantic_color.png (color 버전 사용)
    - 마스크: 주황색 픽셀만 OOD 객체로 처리
    """
    
    def _load_data(self):
        original_dir = os.path.join(self.dataset_dir, 'original')
        label_dir = os.path.join(self.dataset_dir, 'labels')
        
        if not os.path.exists(original_dir) or not os.path.exists(label_dir):
            print(f"Warning: Directory not found - {original_dir} or {label_dir}")
            return
        
        # validation 시리즈의 semantic 라벨 찾기 (color 버전 포함)
        # AnomalyTrack: validation0000_labels_semantic_color.png
        # ObstacleTrack: validation_1_labels_semantic_color.png
        label_patterns = [
            'validation*_labels_semantic_color.png',  # color 버전
            'validation*_labels_semantic.png',        # 일반 버전 (백업)
            'validation_*_labels_semantic_color.png', # ObstacleTrack color 버전
            'validation_*_labels_semantic.png'        # ObstacleTrack 일반 버전 (백업)
        ]
        
        label_files = []
        for pattern in label_patterns:
            found_labels = glob(os.path.join(label_dir, pattern))
            label_files.extend(found_labels)
        
        # 중복 제거 (같은 base name의 경우 color 버전 우선)
        label_dict = {}
        for label_path in label_files:
            label_filename = os.path.basename(label_path)
            
            # base name 추출
            if '_labels_semantic_color' in label_filename:
                base_name = label_filename.replace('_labels_semantic_color.png', '')
                priority = 1  # color 버전이 우선
            elif '_labels_semantic' in label_filename:
                base_name = label_filename.replace('_labels_semantic.png', '')
                priority = 2  # 일반 버전은 백업
            else:
                continue
            
            # 우선순위가 높은 것만 유지
            if base_name not in label_dict or priority < label_dict[base_name][1]:
                label_dict[base_name] = (label_path, priority)
        
        print(f"Found {len(label_dict)} unique label files")
        
        # 이미지-라벨 매칭
        for base_name, (label_path, _) in label_dict.items():
            # 해당하는 이미지 파일 찾기 (다양한 확장자 시도)
            possible_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            image_path = None
            
            for ext in possible_extensions:
                potential_image = os.path.join(original_dir, f"{base_name}{ext}")
                if os.path.exists(potential_image):
                    image_path = potential_image
                    break
            
            if image_path:
                self.image_paths.append(image_path)
                self.label_paths.append(label_path)
            else:
                print(f"Warning: No image found for {base_name}")
    
    def _process_mask(self, mask):
        """Segment Me: 주황색 픽셀만 positive로 처리"""
        if len(mask.shape) == 3:
            # RGB 마스크에서 주황색 픽셀 감지
            
            # 방법 1: RGB 기반 주황색 감지 (관대한 범위)
            # 주황색 범위: R > 150, G > 30, G < 200, B < 100
            orange_mask_rgb = (mask[:, :, 0] > 150) & \
                              (mask[:, :, 1] > 30) & \
                              (mask[:, :, 1] < 200) & \
                              (mask[:, :, 2] < 100)
            
            # 방법 2: HSV 기반 주황색 감지 (더 정확함)
            try:
                import cv2
                hsv = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)
                
                # 주황색 HSV 범위 (OpenCV HSV: H=0-179, S=0-255, V=0-255)
                # 주황색: H=10-25, S=100-255, V=100-255
                lower_orange = np.array([5, 50, 50])    # 더 넓은 범위
                upper_orange = np.array([30, 255, 255])
                
                hsv_orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
                orange_mask_hsv = (hsv_orange_mask > 0)
                
                # 더 많은 픽셀을 찾는 방법 사용
                if np.sum(orange_mask_hsv) > np.sum(orange_mask_rgb):
                    binary_mask = orange_mask_hsv.astype(bool)
                else:
                    binary_mask = orange_mask_rgb.astype(bool)
                    
            except ImportError:
                print("Warning: OpenCV not available, using RGB method only")
                binary_mask = orange_mask_rgb.astype(bool)
                
        else:
            # 그레이스케일 마스크인 경우 (예상되지 않음)
            print("Warning: Grayscale mask detected in Segment Me dataset")
            binary_mask = (mask > 0).astype(bool)
        
        return binary_mask


class DatasetFactory:
    """데이터셋 팩토리 클래스"""
    
    @staticmethod
    def create_dataset(dataset_dir, dataset_type=None):
        """데이터셋 타입에 따라 적절한 데이터셋 클래스 생성"""
        
        if dataset_type is None:
            # 경로를 보고 자동으로 타입 감지
            dataset_type = DatasetFactory._detect_dataset_type(dataset_dir)
        
        print(f"Creating dataset: {dataset_type} for {dataset_dir}")
        
        if dataset_type == 'road_anomaly':
            return RoadAnomalyDataset(dataset_dir)
        elif dataset_type == 'fishyscapes':
            return FishyscapesDataset(dataset_dir)
        elif dataset_type == 'segment_me':
            return SegmentMeDataset(dataset_dir)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    @staticmethod
    def _detect_dataset_type(dataset_dir):
        """디렉토리 경로를 보고 데이터셋 타입 자동 감지"""
        dataset_dir_lower = dataset_dir.lower()
        
        if 'road_anomaly' in dataset_dir_lower:
            return 'road_anomaly'
        elif 'fishyscapes' in dataset_dir_lower:
            return 'fishyscapes'
        elif 'segment_me' in dataset_dir_lower or 'anomalytrack' in dataset_dir_lower or 'obstacletrack' in dataset_dir_lower:
            return 'segment_me'
        else:
            # 기본값으로 road_anomaly 스타일 사용
            print(f"Warning: Could not detect dataset type for {dataset_dir}, using road_anomaly as default")
            return 'road_anomaly'


def test_segment_me_datasets():
    """Segment Me 데이터셋 테스트"""
    dataset_paths = [
        "/home/jeonghyo/Gaebal/Datasets/segment_me_val/dataset_AnomalyTrack",
        "/home/jeonghyo/Gaebal/Datasets/segment_me_val/dataset_ObstacleTrack"
    ]
    
    for dataset_dir in dataset_paths:
        if not os.path.exists(dataset_dir):
            print(f"Skipping {dataset_dir} (not found)")
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing dataset: {dataset_dir}")
        print(f"{'='*60}")
        
        try:
            # 데이터셋 생성
            dataset = DatasetFactory.create_dataset(dataset_dir)
            
            if len(dataset) == 0:
                print("❌ No valid samples found!")
                continue
            
            print(f"✅ Dataset loaded successfully: {len(dataset)} samples")
            
            # 첫 3개 샘플 테스트
            for i in range(min(3, len(dataset))):
                print(f"\n--- Sample {i} ---")
                sample_info = dataset.get_sample_info(i)
                
                print(f"Name: {sample_info['image_name']}")
                print(f"Image size: {sample_info['image_size']}")
                print(f"Mask shape: {sample_info['mask_shape']}")
                print(f"Positive pixels: {sample_info['positive_pixels']}")
                print(f"Positive ratio: {sample_info['positive_ratio']:.4f}")
                
                # 실제 샘플 로딩 테스트
                sample = dataset[i]
                
                # 크기 일치 확인
                image_hw = sample['image'].size[::-1]  # PIL: (W, H) -> (H, W)
                mask_hw = sample['mask'].shape
                
                if image_hw == mask_hw:
                    print(f"✅ Image-mask size match: {image_hw}")
                else:
                    print(f"❌ Size mismatch: Image {image_hw} vs Mask {mask_hw}")
                
                # 마스크 값 확인
                mask_unique = np.unique(sample['mask'])
                print(f"Mask unique values: {mask_unique}")
                
                if len(mask_unique) == 2 and set(mask_unique) == {False, True}:
                    print("✅ Binary mask confirmed")
                elif len(mask_unique) == 1 and mask_unique[0] == False:
                    print("⚠️ No positive pixels found - check orange color detection")
                else:
                    print(f"⚠️ Unexpected mask values: {mask_unique}")
                    
        except Exception as e:
            print(f"❌ Error testing {dataset_dir}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_segment_me_datasets()