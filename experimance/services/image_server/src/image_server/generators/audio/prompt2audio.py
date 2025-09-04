#!/usr/bin/env python3
"""
Prompt-to-Audio Generator for Experimance Image Server.

Generates environmental sound effects from text prompts using TangoFlux,
with intelligent caching, CLAP similarity scoring, and seamless looping.
Follows the experimance audio generator pattern.
"""

import asyncio
import json
import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import soundfile as sf
import torch

from image_server.generators.audio.audio_generator import (
    AudioGenerator, 
    AudioGeneratorCapabilities,
    AudioNormalizer
)
from image_server.generators.audio.audio_config import Prompt2AudioConfig

# Configure logging
logger = logging.getLogger(__name__)

# Constants
SR = 44100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LazyImports:
    """Lazy loading of heavy dependencies to avoid startup delays."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self._clap_model = None
        self._clap_processor = None
        self._tangoflux = None
        self._bge_model = None
        self._imports_loaded = False
        self._cache_dir = cache_dir
        
        # Set up HuggingFace cache if cache_dir is provided
        if self._cache_dir:
            self._setup_hf_cache()
    
    def _setup_hf_cache(self):
        """Configure HuggingFace cache location."""
        if self._cache_dir:
            hf_cache = self._cache_dir / "huggingface_cache"
            hf_cache.mkdir(parents=True, exist_ok=True)
            
            # Set HuggingFace environment variables
            os.environ['TRANSFORMERS_CACHE'] = str(hf_cache)
            os.environ['HF_HOME'] = str(hf_cache)
            os.environ['HUGGINGFACE_HUB_CACHE'] = str(hf_cache)

    def load_clap(self, model_id: str):
        """Load CLAP model and processor."""
        if self._clap_model is None:
            from transformers.models.clap.modeling_clap import ClapModel
            from transformers.models.clap.processing_clap import ClapProcessor
            self._clap_processor = ClapProcessor.from_pretrained(model_id)
            self._clap_model = ClapModel.from_pretrained(model_id)
            device = torch.device(DEVICE)
            self._clap_model = self._clap_model.to(device)
            self._clap_model = self._clap_model.eval()
            logger.info(f"Loaded CLAP model: {model_id}")
        return self._clap_model, self._clap_processor

    def load_tangoflux(self, model_name: str):
        """Load TangoFlux inference model.""" 
        if self._tangoflux is None:
            try:
                from tangoflux import TangoFluxInference
                self._tangoflux = TangoFluxInference(name=model_name, device=DEVICE)
                logger.info(f"Loaded TangoFlux model: {model_name}")
            except ImportError as e:
                raise RuntimeError(f"TangoFlux not installed: {e}")
        return self._tangoflux

    def load_bge(self, model_id: str):
        """Load BGE sentence transformer model."""
        if self._bge_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._bge_model = SentenceTransformer(model_id, device=DEVICE)
                logger.info(f"Loaded BGE model: {model_id}")
            except ImportError as e:
                logger.warning(f"BGE not available: {e}")
                self._bge_model = False  # Mark as unavailable
        return self._bge_model if self._bge_model is not False else None


# Global lazy imports instance - will be initialized with models directory when generator starts
_lazy = None


def get_lazy_imports(models_dir: Optional[Path] = None) -> LazyImports:
    """Get or create the global lazy imports instance."""
    global _lazy
    if _lazy is None:
        audio_models_dir = None
        if models_dir:
            audio_models_dir = models_dir / "audio"
        _lazy = LazyImports(cache_dir=audio_models_dir)
    return _lazy


def normalize_prompt(prompt: str) -> str:
    """Normalize prompt for grouping and comparison."""
    return " ".join(prompt.lower().split())


def normalize_filename(text: str, max_length: int = 40) -> str:
    """Normalize text for use in filenames."""
    import re
    # Convert to lowercase and replace spaces with underscores
    normalized = text.lower().replace(" ", "_")
    # Remove non-alphanumeric characters except underscores and hyphens
    normalized = re.sub(r'[^a-z0-9_\-]', '', normalized)
    # Remove multiple consecutive underscores
    normalized = re.sub(r'_+', '_', normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    # Truncate if too long
    return normalized[:max_length] if len(normalized) > max_length else normalized


@dataclass
class CacheItem:
    """Represents a cached audio item."""
    path: str
    prompt: str
    prompt_norm: str
    duration_s: float
    clap_similarity: Optional[float] = None
    timestamp: float = 0.0


class AudioSemanticCache:
    """Lightweight semantic cache for audio files with CLAP and BGE embeddings."""
    
    def __init__(self, cache_dir: str = "audio_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        
        # File paths
        self.meta_path = self.cache_dir / "meta.jsonl"
        self.clap_text_path = self.cache_dir / "clap_text.npy"
        self.clap_audio_path = self.cache_dir / "clap_audio.npy"
        self.bge_text_path = self.cache_dir / "bge_text.npy"
        
        # In-memory data
        self.items: List[CacheItem] = []
        self.clap_text_embeddings: Optional[np.ndarray] = None
        self.clap_audio_embeddings: Optional[np.ndarray] = None
        self.bge_text_embeddings: Optional[np.ndarray] = None
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self):
        """Load existing cache from disk."""
        # Load metadata
        if self.meta_path.exists():
            with open(self.meta_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    # Convert old format if needed
                    if 'ts' in data:
                        data['timestamp'] = data.pop('ts')
                    if 'dur' in data:
                        data['duration_s'] = data.pop('dur')
                    self.items.append(CacheItem(**data))
        
        # Load embeddings
        if self.clap_text_path.exists():
            self.clap_text_embeddings = np.load(self.clap_text_path)
        if self.clap_audio_path.exists():
            self.clap_audio_embeddings = np.load(self.clap_audio_path)
        if self.bge_text_path.exists():
            self.bge_text_embeddings = np.load(self.bge_text_path)
        
        logger.info(f"Loaded {len(self.items)} items from audio cache")

    def _append_embedding(self, file_path: Path, embedding: np.ndarray):
        """Append embedding to existing numpy file or create new one."""
        embedding = embedding.astype(np.float32).reshape(1, -1)
        
        if file_path.exists():
            existing = np.load(file_path)
            combined = np.vstack([existing, embedding])
            np.save(file_path, combined)
        else:
            np.save(file_path, embedding)

    def add_item(self, audio_path: str, prompt: str, clap_model, clap_processor, bge_model=None):
        """Add new item to cache with embeddings."""
        with self.lock:
            # Create cache item
            item = CacheItem(
                path=audio_path,
                prompt=prompt,
                prompt_norm=normalize_prompt(prompt),
                duration_s=self._get_audio_duration(audio_path),
                timestamp=time.time()
            )
            
            # Generate embeddings
            clap_text_emb = self._generate_clap_text_embedding(prompt, clap_model, clap_processor)
            clap_audio_emb = self._generate_clap_audio_embedding(audio_path, clap_model, clap_processor)
            
            # Calculate similarity score
            item.clap_similarity = float(np.dot(clap_text_emb, clap_audio_emb))
            
            # Add to metadata
            with open(self.meta_path, 'a') as f:
                f.write(json.dumps(item.__dict__) + '\n')
            self.items.append(item)
            
            # Add embeddings
            self._append_embedding(self.clap_text_path, clap_text_emb)
            self._append_embedding(self.clap_audio_path, clap_audio_emb)
            
            if bge_model:
                bge_emb = self._generate_bge_embedding(prompt, bge_model)
                self._append_embedding(self.bge_text_path, bge_emb)
            
            # Reload embeddings
            self._load_cache()
            
            logger.debug(f"Added item to cache: {audio_path} (similarity: {item.clap_similarity:.3f})")

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file."""
        try:
            with sf.SoundFile(audio_path) as f:
                return len(f) / f.samplerate
        except Exception:
            return 0.0

    @torch.no_grad()
    def _generate_clap_text_embedding(self, text: str, model, processor) -> np.ndarray:
        """Generate CLAP text embedding."""
        inputs = processor(text=[text], return_tensors="pt").to(DEVICE)
        embedding = model.get_text_features(**inputs)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding.squeeze(0).float().cpu().numpy()

    @torch.no_grad()
    def _generate_clap_audio_embedding(self, audio_path: str, model, processor) -> np.ndarray:
        """Generate CLAP audio embedding with multi-window averaging."""
        try:
            # Load audio
            audio_data, sr = sf.read(audio_path)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono
            
            # Resample to CLAP's expected 48kHz if needed
            CLAP_SR = 48000
            if sr != CLAP_SR:
                import torchaudio.transforms as T
                resampler = T.Resample(sr, CLAP_SR)
                audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
                audio_data = resampler(audio_tensor).numpy()
            
            # Multi-window embedding for long audio
            CLAP_SR = 48000
            window_samples = min(len(audio_data), 10 * CLAP_SR)  # 10 second windows at CLAP sample rate
            if len(audio_data) > window_samples:
                # Start, middle, end windows
                starts = [0, max(0, (len(audio_data) - window_samples) // 2), max(0, len(audio_data) - window_samples)]
                embeddings = []
                for start in starts:
                    segment = audio_data[start:start + window_samples]
                    if len(segment) < window_samples:
                        # Pad if necessary
                        segment = np.pad(segment, (0, window_samples - len(segment)))
                    
                    inputs = processor(audios=segment, sampling_rate=CLAP_SR, return_tensors="pt").to(DEVICE)
                    embedding = model.get_audio_features(**inputs)
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
                    embeddings.append(embedding)
                
                # Average embeddings
                final_embedding = torch.mean(torch.stack(embeddings), dim=0)
            else:
                # Single window
                inputs = processor(audios=audio_data, sampling_rate=CLAP_SR, return_tensors="pt").to(DEVICE)
                final_embedding = model.get_audio_features(**inputs)
                final_embedding = torch.nn.functional.normalize(final_embedding, p=2, dim=-1)
            
            return final_embedding.squeeze(0).float().cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error generating CLAP audio embedding for {audio_path}: {e}")
            # Return zero embedding as fallback
            return np.zeros(512, dtype=np.float32)

    def _generate_bge_embedding(self, text: str, bge_model) -> np.ndarray:
        """Generate BGE text embedding."""
        if bge_model is None:
            return np.zeros(384, dtype=np.float32)  # BGE-small dimension
        return bge_model.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    def find_exact_matches(self, prompt: str) -> List[int]:
        """Find exact prompt matches."""
        norm_prompt = normalize_prompt(prompt)
        return [i for i, item in enumerate(self.items) if item.prompt_norm == norm_prompt]

    def find_semantic_matches(self, prompt: str, threshold: float = 0.70, max_results: int = 10) -> List[int]:
        """Find semantic matches using BGE or CLAP text embeddings."""
        if len(self.items) == 0:
            return []
        
        # Prefer BGE for text-to-text matching
        if self.bge_text_embeddings is not None:
            bge_model = _lazy.load_bge("BAAI/bge-small-en-v1.5")  # Default BGE model
            if bge_model:
                query_emb = self._generate_bge_embedding(prompt, bge_model)
                similarities = self.bge_text_embeddings @ query_emb
                matches = np.where(similarities >= threshold)[0]
                return matches[np.argsort(-similarities[matches])][:max_results].tolist()
        
        # Fallback to CLAP text embeddings
        if self.clap_text_embeddings is not None:
            clap_model, clap_processor = _lazy.load_clap("laion/clap-htsat-unfused")
            query_emb = self._generate_clap_text_embedding(prompt, clap_model, clap_processor)
            similarities = self.clap_text_embeddings @ query_emb
            matches = np.where(similarities >= threshold)[0]
            return matches[np.argsort(-similarities[matches])][:max_results].tolist()
        
        return []

    def select_from_candidates(
        self,
        prompt: str,
        candidate_indices: List[int],
        threshold: float = 0.35,
        temperature: float = 0.25,
        duration_range: Tuple[float, float] = (10, 40)
    ) -> Optional[Dict[str, Any]]:
        """Select best candidate using CLAP audio-text similarity with weighted random selection."""
        if not candidate_indices or self.clap_audio_embeddings is None:
            return None
        
        # Generate query embedding
        clap_model, clap_processor = _lazy.load_clap("laion/clap-htsat-unfused")
        query_emb = self._generate_clap_text_embedding(prompt, clap_model, clap_processor)
        
        # Filter candidates
        valid_candidates = []
        similarities = []
        
        for idx in candidate_indices:
            if idx >= len(self.items):
                logger.debug(f"Candidate index {idx} exceeds items length {len(self.items)}, skipping")
                continue
                
            # Check bounds for embeddings array
            if self.clap_audio_embeddings is not None and idx >= len(self.clap_audio_embeddings):
                logger.warning(f"Candidate index {idx} exceeds audio embeddings length {len(self.clap_audio_embeddings)}, skipping")
                continue
                
            item = self.items[idx]
            if not (duration_range[0] <= item.duration_s <= duration_range[1]):
                continue
            
            # Calculate similarity with query - with additional safety check
            try:
                if self.clap_audio_embeddings is None:
                    logger.warning("CLAP audio embeddings not available")
                    continue
                    
                similarity = float(np.dot(self.clap_audio_embeddings[idx], query_emb))
                if similarity >= threshold:
                    valid_candidates.append(idx)
                    similarities.append(similarity)
            except Exception as e:
                logger.error(f"Error calculating similarity for index {idx}: {e}")
                continue
        
        if not valid_candidates:
            return None
        
        # Weighted random selection
        similarities = np.array(similarities, dtype=np.float32)
        if temperature <= 0:
            # Deterministic: pick best
            best_idx = np.argmax(similarities)
            selected_idx = valid_candidates[best_idx]
            selected_similarity = similarities[best_idx]
        else:
            # Weighted random using softmax
            logits = similarities / max(1e-6, temperature)
            logits = logits - logits.max()  # Numerical stability
            probs = np.exp(logits)
            probs = probs / probs.sum()
            
            choice = np.random.choice(len(valid_candidates), p=probs)
            selected_idx = valid_candidates[choice]
            selected_similarity = similarities[choice]
        
        return {
            "index": selected_idx,
            "item": self.items[selected_idx],
            "similarity": float(selected_similarity),
            "pool_size": len(valid_candidates)
        }

    def prune_prompt_variants(self, prompt: str, max_variants: int = 5, strategy: str = "quality"):
        """Prune variants for a specific prompt."""
        exact_matches = self.find_exact_matches(prompt)
        if len(exact_matches) <= max_variants:
            return
        
        with self.lock:
            if strategy == "quality":
                # Keep highest CLAP similarity scores
                scored_items = [(i, self.items[i].clap_similarity or 0.0) for i in exact_matches]
                keep_indices = [i for i, _ in sorted(scored_items, key=lambda x: x[1], reverse=True)[:max_variants]]
            elif strategy == "diversity":
                # Keep most diverse items (farthest-first selection)
                if self.clap_audio_embeddings is None:
                    keep_indices = exact_matches[:max_variants]  # Fallback
                else:
                    # Validate that all exact_matches indices are within bounds
                    valid_matches = [i for i in exact_matches if i < len(self.clap_audio_embeddings)]
                    if len(valid_matches) != len(exact_matches):
                        logger.warning(f"Some exact matches out of bounds: {len(exact_matches)} -> {len(valid_matches)}")
                    
                    if not valid_matches:
                        keep_indices = exact_matches[:max_variants]  # Fallback
                    else:
                        try:
                            embeddings = self.clap_audio_embeddings[valid_matches]
                            
                            # Start with best quality item
                            scored_items = [(i, self.items[i].clap_similarity or 0.0) for i in valid_matches]
                            best_idx = max(range(len(scored_items)), key=lambda x: scored_items[x][1])
                            selected = [best_idx]
                            
                            # Greedily select most diverse remaining items
                            while len(selected) < max_variants:
                                best_distance = -1
                                best_candidate = -1
                                
                                for i in range(len(valid_matches)):
                                    if i in selected:
                                        continue
                                    
                                    # Calculate minimum distance to already selected items
                                    min_distance = min([
                                        1.0 - float(np.dot(embeddings[i], embeddings[j]))
                                        for j in selected
                                    ])
                                    
                                    if min_distance > best_distance:
                                        best_distance = min_distance
                                        best_candidate = i
                                
                                if best_candidate >= 0:
                                    selected.append(best_candidate)
                                else:
                                    break
                            
                            keep_indices = [valid_matches[i] for i in selected]
                        except Exception as e:
                            logger.error(f"Error in diversity selection: {e}")
                            keep_indices = exact_matches[:max_variants]  # Fallback
            else:
                # Default: keep newest
                keep_indices = sorted(exact_matches, key=lambda i: self.items[i].timestamp, reverse=True)[:max_variants]
            
            # Remove items not in keep_indices
            remove_indices = set(exact_matches) - set(keep_indices)
            if remove_indices:
                self._remove_items(remove_indices)

    def _remove_items(self, remove_indices: set):
        """Remove items from cache and update files."""
        # This is a simplified implementation - in production you'd want to
        # properly rebuild the numpy arrays and jsonl file
        logger.warning(f"Cache pruning not fully implemented - would remove {len(remove_indices)} items")
        # TODO: Implement full cache pruning with array reconstruction

    # Cache Management Methods
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self.items:
            return {
                "total_items": 0,
                "total_size_mb": 0.0,
                "oldest_item": None,
                "newest_item": None,
                "average_duration": 0.0,
                "unique_prompts": 0,
                "average_clap_similarity": 0.0,
                "missing_files": 0
            }
        
        # Calculate total file size
        total_size = 0
        valid_items = []
        for item in self.items:
            if Path(item.path).exists():
                try:
                    total_size += Path(item.path).stat().st_size
                    valid_items.append(item)
                except OSError:
                    continue
        
        # Calculate statistics
        timestamps = [item.timestamp for item in valid_items]
        durations = [item.duration_s for item in valid_items]
        clap_scores = [item.clap_similarity for item in valid_items if item.clap_similarity is not None]
        unique_prompts = len(set(item.prompt_norm for item in valid_items))
        
        return {
            "total_items": len(valid_items),
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_item": min(timestamps) if timestamps else None,
            "newest_item": max(timestamps) if timestamps else None,
            "average_duration": sum(durations) / len(durations) if durations else 0.0,
            "unique_prompts": unique_prompts,
            "average_clap_similarity": sum(clap_scores) / len(clap_scores) if clap_scores else 0.0,
            "missing_files": len(self.items) - len(valid_items)
        }
    
    def list_cache_items(self, limit: int = 20, sort_by: str = "timestamp", reverse: bool = True) -> List[Dict[str, Any]]:
        """List cache items with metadata."""
        if not self.items:
            return []
        
        # Sort items
        if sort_by == "timestamp":
            sorted_items = sorted(self.items, key=lambda x: x.timestamp, reverse=reverse)
        elif sort_by == "clap_similarity":
            sorted_items = sorted(self.items, key=lambda x: x.clap_similarity or 0.0, reverse=reverse)
        elif sort_by == "duration":
            sorted_items = sorted(self.items, key=lambda x: x.duration_s, reverse=reverse)
        elif sort_by == "prompt":
            sorted_items = sorted(self.items, key=lambda x: x.prompt, reverse=reverse)
        else:
            sorted_items = self.items
        
        # Convert to dict format
        result = []
        for item in sorted_items[:limit]:
            file_exists = Path(item.path).exists()
            file_size = 0
            if file_exists:
                try:
                    file_size = Path(item.path).stat().st_size
                except OSError:
                    file_exists = False
            
            result.append({
                "path": item.path,
                "prompt": item.prompt,
                "duration_s": item.duration_s,
                "clap_similarity": item.clap_similarity,
                "timestamp": item.timestamp,
                "age_days": (time.time() - item.timestamp) / 86400,
                "file_exists": file_exists,
                "file_size_kb": file_size / 1024
            })
        
        return result
    
    def clear_cache(self, confirm: bool = False) -> Dict[str, Any]:
        """Clear entire cache."""
        if not confirm:
            return {"error": "Must set confirm=True to clear cache"}
        
        with self.lock:
            # Count items to be removed
            total_items = len(self.items)
            total_size = 0
            removed_files = 0
            
            # Remove audio files
            for item in self.items:
                if Path(item.path).exists():
                    try:
                        file_size = Path(item.path).stat().st_size
                        total_size += file_size
                        Path(item.path).unlink()
                        removed_files += 1
                    except OSError as e:
                        logger.warning(f"Could not remove {item.path}: {e}")
            
            # Remove cache files
            cache_files = [self.meta_path, self.clap_text_path, self.clap_audio_path, self.bge_text_path]
            for cache_file in cache_files:
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except OSError as e:
                        logger.warning(f"Could not remove {cache_file}: {e}")
            
            # Clear in-memory data
            self.items.clear()
            self.clap_text_embeddings = None
            self.clap_audio_embeddings = None
            self.bge_text_embeddings = None
            
            return {
                "removed_items": total_items,
                "removed_files": removed_files,
                "freed_space_mb": total_size / (1024 * 1024)
            }
    
    def remove_old_items(self, days_old: float) -> Dict[str, Any]:
        """Remove cache items older than specified days."""
        cutoff_time = time.time() - (days_old * 86400)
        
        with self.lock:
            old_items = [i for i, item in enumerate(self.items) if item.timestamp < cutoff_time]
            
            if not old_items:
                return {"removed_items": 0, "freed_space_mb": 0.0}
            
            # Remove files
            total_size = 0
            removed_files = 0
            for idx in old_items:
                item = self.items[idx]
                if Path(item.path).exists():
                    try:
                        file_size = Path(item.path).stat().st_size
                        total_size += file_size
                        Path(item.path).unlink()
                        removed_files += 1
                    except OSError as e:
                        logger.warning(f"Could not remove {item.path}: {e}")
            
            # Rebuild cache without old items
            self._rebuild_cache_without_indices(set(old_items))
            
            return {
                "removed_items": len(old_items),
                "removed_files": removed_files,
                "freed_space_mb": total_size / (1024 * 1024)
            }
    
    def remove_by_prompt_pattern(self, pattern: str, confirm: bool = False) -> Dict[str, Any]:
        """Remove cache items matching a prompt pattern."""
        if not confirm:
            return {"error": "Must set confirm=True to remove items"}
        
        import re
        pattern_re = re.compile(pattern, re.IGNORECASE)
        
        with self.lock:
            matching_items = [i for i, item in enumerate(self.items) if pattern_re.search(item.prompt)]
            
            if not matching_items:
                return {"removed_items": 0, "freed_space_mb": 0.0}
            
            # Remove files
            total_size = 0
            removed_files = 0
            for idx in matching_items:
                item = self.items[idx]
                if Path(item.path).exists():
                    try:
                        file_size = Path(item.path).stat().st_size
                        total_size += file_size
                        Path(item.path).unlink()
                        removed_files += 1
                    except OSError as e:
                        logger.warning(f"Could not remove {item.path}: {e}")
            
            # Rebuild cache without matching items
            self._rebuild_cache_without_indices(set(matching_items))
            
            return {
                "removed_items": len(matching_items),
                "removed_files": removed_files,
                "freed_space_mb": total_size / (1024 * 1024),
                "pattern": pattern
            }
    
    def _rebuild_cache_without_indices(self, remove_indices: set):
        """Rebuild cache files without specified indices."""
        # Keep items not in remove_indices
        keep_items = [item for i, item in enumerate(self.items) if i not in remove_indices]
        
        # Rebuild metadata file
        with open(self.meta_path, 'w') as f:
            for item in keep_items:
                f.write(json.dumps(item.__dict__) + '\n')
        
        # Rebuild embedding arrays
        if self.clap_text_embeddings is not None:
            keep_clap_text = np.array([self.clap_text_embeddings[i] for i in range(len(self.items)) if i not in remove_indices])
            np.save(self.clap_text_path, keep_clap_text)
        
        if self.clap_audio_embeddings is not None:
            keep_clap_audio = np.array([self.clap_audio_embeddings[i] for i in range(len(self.items)) if i not in remove_indices])
            np.save(self.clap_audio_path, keep_clap_audio)
        
        if self.bge_text_embeddings is not None:
            keep_bge_text = np.array([self.bge_text_embeddings[i] for i in range(len(self.items)) if i not in remove_indices])
            np.save(self.bge_text_path, keep_bge_text)
        
        # Update in-memory data
        self.items = keep_items
        self._load_cache()  # Reload to refresh embeddings
    
    def find_duplicates(self) -> List[List[Dict[str, Any]]]:
        """Find potential duplicate audio files based on prompts and similarity."""
        if not self.items:
            return []
        
        # Group by normalized prompt
        prompt_groups = {}
        for i, item in enumerate(self.items):
            norm_prompt = item.prompt_norm
            if norm_prompt not in prompt_groups:
                prompt_groups[norm_prompt] = []
            prompt_groups[norm_prompt].append({
                "index": i,
                "item": item,
                "path": item.path,
                "prompt": item.prompt,
                "clap_similarity": item.clap_similarity,
                "timestamp": item.timestamp,
                "file_exists": Path(item.path).exists()
            })
        
        # Return groups with multiple items
        duplicates = [group for group in prompt_groups.values() if len(group) > 1]
        
        # Sort each group by timestamp (newest first)
        for group in duplicates:
            group.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Sort groups by group size (largest first)
        duplicates.sort(key=len, reverse=True)
        
        return duplicates


class Prompt2AudioGenerator(AudioGenerator):
    """TangoFlux-based audio generator with semantic caching and CLAP scoring."""
    
    supported_capabilities = {
        AudioGeneratorCapabilities.TEXT_TO_AUDIO,
        AudioGeneratorCapabilities.ENVIRONMENTAL_SOUNDS,
        AudioGeneratorCapabilities.SEAMLESS_LOOPS,
        AudioGeneratorCapabilities.CUSTOM_DURATION,
        AudioGeneratorCapabilities.SEMANTIC_CACHING,
        AudioGeneratorCapabilities.LOUDNESS_NORMALIZATION,
        AudioGeneratorCapabilities.CLAP_SCORING,
        AudioGeneratorCapabilities.VARIANT_GENERATION,
        AudioGeneratorCapabilities.BACKGROUND_PREFETCH
    }
    
    def _configure(self, config: Prompt2AudioConfig, **kwargs):
        """Configure prompt2audio generator settings."""
        self.config = Prompt2AudioConfig(**{
            **config.model_dump(),
            **kwargs
        })
        
        # Initialize lazy imports with models directory
        self._lazy = get_lazy_imports(self.config.models_dir)
        
        # Initialize cache
        self.cache = AudioSemanticCache(str(self.config.cache_dir))

        logger.info(f"Prompt2AudioGenerator configured with cache: {self.config.cache_dir}")
        logger.info(f"Models will be stored in: {self.config.models_dir / 'audio'}")

    async def _generate_audio_impl(self, prompt: str, **kwargs) -> str:
        """Generate audio from text prompt with caching and quality scoring."""
        self._validate_prompt(prompt)
        
        # Override config with kwargs
        duration_s = kwargs.get('duration_s', self.config.duration_s)
        force_generation = kwargs.get('force_generation', False)
        
        logger.info(f"Generating audio for prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}' (force: {force_generation})")
        
        # Try cache lookup first (unless forced)
        if not force_generation:
            result = await self._try_cache_lookup(prompt, duration_s)
            if result:
                # Store comprehensive metadata from cache hit
                cache_item = result["item"]
                self._last_generation_metadata = {
                    # From cache item
                    'clap_similarity': cache_item.clap_similarity,
                    'duration_s': cache_item.duration_s,
                    'cached_timestamp': cache_item.timestamp,
                    
                    # Cache-specific info
                    'cached': True,
                    'cache_type': result.get('match_type', 'unknown'),
                    'cache_similarity': result.get('similarity'),  # Search similarity, different from clap_similarity
                    
                    # Current generation config (what would be used if generating new)
                    'is_loop': self.config.enable_seamless_loop,
                    'model': self.config.model_name,
                    'steps': self.config.steps,
                    'guidance_scale': self.config.guidance_scale,
                    'requested_duration_s': duration_s,
                    
                    # Additional cache metadata
                    'cache_age_seconds': time.time() - cache_item.timestamp
                }
                
                # Optionally prefetch more variants
                if self.config.prefetch_in_background:
                    threading.Thread(
                        target=self._prefetch_variants,
                        args=(prompt,),
                        daemon=True
                    ).start()
                elif self.config.prefetch_new:
                    self._prefetch_variants(prompt)
                
                return result["item"].path
        
        # Generate new audio
        result = await self._generate_new_audio(prompt, duration_s)
        
        return result

    async def _try_cache_lookup(self, prompt: str, duration_s: int) -> Optional[Dict[str, Any]]:
        """Try to find suitable audio in cache."""
        duration_range = (duration_s * 0.8, duration_s * 1.2)
        
        # 1. Try exact prompt matches
        exact_matches = self.cache.find_exact_matches(prompt)
        result = self.cache.select_from_candidates(
            prompt, exact_matches,
            threshold=self.config.tau_use,
            temperature=self.config.temperature,
            duration_range=duration_range
        )
        if result:
            logger.info(f"Cache hit (exact): {result['item'].path} (similarity: {result['similarity']:.3f})")
            result['match_type'] = 'exact'
            return result
        
        # 2. Try semantic matches
        semantic_matches = self.cache.find_semantic_matches(
            prompt, 
            threshold=self.config.tau_prompt_sem,
            max_results=self.config.reuse_k
        )
        result = self.cache.select_from_candidates(
            prompt, semantic_matches,
            threshold=self.config.tau_use,
            temperature=self.config.temperature,
            duration_range=duration_range
        )
        if result:
            logger.info(f"Cache hit (semantic): {result['item'].path} (similarity: {result['similarity']:.3f})")
            result['match_type'] = 'semantic'
            return result
        
        # 3. Try global fallback
        all_indices = list(range(len(self.cache.items)))
        result = self.cache.select_from_candidates(
            prompt, all_indices,
            threshold=max(0.2, self.config.tau_use - 0.1),  # Lower threshold for fallback
            temperature=self.config.temperature,
            duration_range=duration_range
        )
        if result:
            logger.info(f"Cache hit (global): {result['item'].path} (similarity: {result['similarity']:.3f})")
            result['match_type'] = 'global'
            return result
        
        return None

    async def _generate_new_audio(self, prompt: str, duration_s: int) -> str:
        """Generate new audio using TangoFlux."""
        logger.info(f"Generating new audio: {self.config.candidates} candidates, {duration_s}s duration")
        logger.debug(f"TangoFlux config - steps: {self.config.steps}, guidance_scale: {self.config.guidance_scale}")
        logger.debug(f"Prompt length: {len(prompt)}, duration: {duration_s}s")
        
        # Load models
        tangoflux = self._lazy.load_tangoflux(self.config.model_name)
        clap_model, clap_processor = self._lazy.load_clap(self.config.clap_model)
        
        # Generate candidates
        candidates = []
        for i in range(self.config.candidates):
            logger.debug(f"Generating candidate {i+1}/{self.config.candidates}")
            try:
                audio = tangoflux.generate(
                    prompt,
                    steps=self.config.steps,
                    duration=duration_s,
                    guidance_scale=self.config.guidance_scale
                )
                logger.debug(f"Candidate {i+1} generated successfully, type: {type(audio)}")
                if hasattr(audio, 'shape'):
                    logger.debug(f"Candidate {i+1} shape: {audio.shape}")
            except Exception as e:
                logger.error(f"Error generating candidate {i+1}: {e}")
                raise
            
            # Convert to tensor if needed
            if not isinstance(audio, torch.Tensor):
                logger.debug(f"Converting candidate {i+1} from {type(audio)} to tensor")
                try:
                    audio = torch.from_numpy(audio)
                    logger.debug(f"Conversion successful, tensor shape: {audio.shape}")
                except Exception as e:
                    logger.error(f"Error converting candidate {i+1} to tensor: {e}")
                    raise
            
            candidates.append(audio.float().contiguous())
            logger.debug(f"Candidate {i+1} added to candidates list")
        
        logger.info(f"Successfully generated {len(candidates)} candidates")
        
        # Validate candidates before processing
        if not candidates:
            raise RuntimeError("No candidates were generated successfully")
        
        logger.debug(f"Processing {len(candidates)} candidates for CLAP scoring")
        
        # Process candidates and select best
        accepted_candidates = []
        query_emb = self.cache._generate_clap_text_embedding(prompt, clap_model, clap_processor)
        
        for i, audio_tensor in enumerate(candidates):
            logger.debug(f"Processing candidate {i+1}/{len(candidates)}")
            
            # Validate candidate tensor
            if audio_tensor is None:
                logger.warning(f"Candidate {i+1} is None, skipping")
                continue
                
            if not isinstance(audio_tensor, torch.Tensor):
                logger.warning(f"Candidate {i+1} is not a tensor ({type(audio_tensor)}), skipping")
                continue
                
            if audio_tensor.numel() == 0:
                logger.warning(f"Candidate {i+1} is empty tensor, skipping")
                continue
            
            try:
                # Make seamless loop
                if self.config.enable_seamless_loop:
                    # Adaptive crossfade duration: use configured duration but cap at 25% of audio length
                    max_crossfade = duration_s * 0.25  # Max 25% of audio length
                    adaptive_crossfade = min(self.config.tail_duration_s, max_crossfade)
                    looped_audio = AudioNormalizer.make_seamless_loop(
                        audio_tensor, SR, adaptive_crossfade
                    )
                else:
                    looped_audio = audio_tensor
                
                # Calculate CLAP similarity
                audio_np = looped_audio.cpu().numpy()
                if looped_audio.ndim > 1:
                    audio_np = audio_np.mean(axis=0)  # Convert to mono for CLAP
                
                # Generate temp file for CLAP scoring
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                sf.write(temp_path, audio_np, SR, subtype='PCM_16')
                
                try:
                    audio_emb = self.cache._generate_clap_audio_embedding(temp_path, clap_model, clap_processor)
                    similarity = float(np.dot(query_emb, audio_emb))
                    
                    if similarity >= self.config.tau_accept_new:
                        accepted_candidates.append((similarity, looped_audio, audio_np))
                        logger.debug(f"Candidate {i+1} accepted (similarity: {similarity:.3f})")
                    else:
                        logger.debug(f"Candidate {i+1} rejected (similarity: {similarity:.3f})")
                finally:
                    os.unlink(temp_path)
                    
            except Exception as e:
                logger.error(f"Error processing candidate {i+1}: {e}")
                continue
        
        # Handle results
        if not accepted_candidates:
            # Fallback: return best candidate as temp file
            if not candidates:
                raise RuntimeError("No candidates were generated and no fallback available")
                
            best_idx = 0  # Could implement better selection here
            if best_idx >= len(candidates):
                raise RuntimeError(f"Fallback index {best_idx} out of bounds for {len(candidates)} candidates")
                
            try:
                fallback_candidate = candidates[best_idx]
                audio_np = fallback_candidate.cpu().numpy()
                if fallback_candidate.ndim > 1:
                    audio_np = audio_np.mean(axis=0)
                
                temp_path = self.output_dir / f"temp_{normalize_filename(prompt)}_{int(time.time() * 1000)}.wav"
                sf.write(str(temp_path), audio_np, SR, subtype='PCM_16')
                logger.warning(f"No candidates accepted, returning temp file: {temp_path}")
                return str(temp_path)
            except Exception as e:
                logger.error(f"Error creating fallback audio file: {e}")
                raise RuntimeError(f"Failed to create fallback audio: {e}")
        
        # Select from accepted candidates (weighted random or best)
        similarities = [c[0] for c in accepted_candidates]
        if self.config.temperature <= 0:
            # Pick best
            best_idx = np.argmax(similarities)
        else:
            # Weighted random
            sims = np.array(similarities, dtype=np.float32)
            logits = sims / max(1e-6, self.config.temperature)
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            best_idx = np.random.choice(len(accepted_candidates), p=probs)
        
        # Bounds check for selected index
        if best_idx >= len(accepted_candidates):
            logger.error(f"Selected index {best_idx} out of bounds for {len(accepted_candidates)} accepted candidates")
            best_idx = 0  # Fallback to first candidate
        
        similarity, looped_audio, audio_np = accepted_candidates[best_idx]
        
        # Save final audio
        filename = f"{normalize_filename(prompt)}_{int(time.time() * 1000)}.wav"
        audio_path = self.output_dir / filename
        sf.write(str(audio_path), audio_np, SR, subtype='PCM_16')
        
        # Apply loudness normalization if enabled
        if self.config.normalize_loudness:
            try:
                normalized_path = AudioNormalizer.normalize_loudness(
                    str(audio_path),
                    target_lufs=self.config.target_lufs,
                    true_peak_dbfs=self.config.true_peak_dbfs
                )
                # Replace original with normalized version
                os.rename(normalized_path, str(audio_path))
                logger.debug(f"Applied loudness normalization to {audio_path}")
            except Exception as e:
                logger.warning(f"Loudness normalization failed: {e}")
        
        # Add to cache
        bge_model = self._lazy.load_bge(self.config.bge_model) if self.config.use_bge else None
        self.cache.add_item(str(audio_path), prompt, clap_model, clap_processor, bge_model)
        
        # Prune cache if needed
        self.cache.prune_prompt_variants(
            prompt, 
            max_variants=self.config.max_per_prompt,
            strategy=self.config.cap_strategy
        )
        
        logger.info(f"Generated and cached new audio: {audio_path} (similarity: {similarity:.3f})")
        
        # Store metadata for potential retrieval
        metadata_to_store = {
            # Generation results
            'clap_similarity': similarity,
            'duration_s': duration_s,
            
            # Generation info
            'cached': False,
            'cache_type': 'new_generation',
            'generation_timestamp': time.time(),
            
            # Generation config used
            'is_loop': self.config.enable_seamless_loop,
            'model': self.config.model_name,
            'steps': self.config.steps,
            'guidance_scale': self.config.guidance_scale,
            'requested_duration_s': duration_s,
            
            # Model parameters
            'candidates': self.config.candidates,
            'tau_accept_new': self.config.tau_accept_new
        }
        
        self._last_generation_metadata = metadata_to_store
        
        return str(audio_path)

    def _prefetch_variants(self, prompt: str):
        """Prefetch additional variants for a prompt."""
        try:
            existing_count = len(self.cache.find_exact_matches(prompt))
            needed = max(0, self.config.target_per_prompt - existing_count)
            needed = min(needed, self.config.max_new_when_prefetch)
            
            if needed <= 0:
                return
            
            logger.info(f"Prefetching {needed} variants for prompt: '{prompt[:50]}'")
            
            # Run the async prefetch in a new event loop since we're in a thread
            asyncio.run(self._generate_prefetch_variants(prompt, needed))
            
        except Exception as e:
            logger.error(f"Error during prefetch: {e}")

    async def _generate_prefetch_variants(self, prompt: str, count: int):
        """Generate additional variants for prefetching."""
        try:
            for _ in range(count):
                await self._generate_new_audio(prompt, self.config.duration_s)
        except Exception as e:
            logger.error(f"Error generating prefetch variants: {e}")

    async def start(self):
        """Start the generator and pre-warm models if configured."""
        await super().start()
        
        # Initialize metadata storage
        self._last_generation_metadata = {}
        
        if self.config.pre_warm:
            logger.info("Pre-warming Prompt2AudioGenerator models...")
            try:
                # Load all models to GPU
                self._lazy.load_tangoflux(self.config.model_name)
                self._lazy.load_clap(self.config.clap_model)
                if self.config.use_bge:
                    self._lazy.load_bge(self.config.bge_model)
                logger.info("Model pre-warming completed")
            except Exception as e:
                logger.error(f"Model pre-warming failed: {e}")
    
    def get_last_generation_metadata(self) -> dict:
        """Get metadata from the last generation."""
        return getattr(self, '_last_generation_metadata', {})
