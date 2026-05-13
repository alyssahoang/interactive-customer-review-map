"""Core package for P15 semantic-geometry experiments."""

from .clusterer import ClusteringEngine
from .data_loader import OlistReviewLoader
from .embedder import EmbeddingFactory
from .experiment_config import ExperimentConfig
from .evaluator import GeometryEvaluator
from .interpretability import SemanticInspector
from .notebook_phase3_helpers import GeometryClusteringEvaluator, SemanticEmbedder
from .preprocessor import PortugueseSentimentLabeler, PortugueseTextPreprocessor
from .projection import ProjectionEngine
from .quality import DataQualityAuditor
from .reproducibility import ArtifactStore, set_global_seed
from .stability import ClusterStabilityAnalyzer
from .submission_pipeline import SubmissionPipeline
from .topic_modeler import ClusterTopicModeler

__all__ = [
    "ArtifactStore",
    "ClusteringEngine",
    "ClusterStabilityAnalyzer",
    "ClusterTopicModeler",
    "DataQualityAuditor",
    "EmbeddingFactory",
    "ExperimentConfig",
    "GeometryEvaluator",
    "GeometryClusteringEvaluator",
    "OlistReviewLoader",
    "PortugueseSentimentLabeler",
    "PortugueseTextPreprocessor",
    "ProjectionEngine",
    "SemanticEmbedder",
    "SemanticInspector",
    "SubmissionPipeline",
    "set_global_seed",
]
