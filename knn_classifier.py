"""
knn_classifier.py — K-Nearest Neighbors Algorithm (from scratch!)

WHAT IS KNN?
    Imagine you're new at school. You don't know anyone.
    You look at the 3 people sitting closest to you.
    2 of them are wearing blue shirts, 1 is wearing red.
    You guess: "I'm probably in the BLUE team."
    
    That's KNN! Find the K nearest neighbors, see what most of them are,
    and that's your answer.

HOW IT WORKS HERE:
    1. We have training images with labels ("car", "bike", etc.)
    2. Each image is converted to a list of numbers (features)
    3. When a new image comes in, we convert it to numbers too
    4. We find the K training images whose numbers are most similar
    5. We see what most of them are labeled as → THAT'S OUR ANSWER!
"""

import math
from collections import Counter


class KNNClassifier:
    """
    K-Nearest Neighbors classifier.
    
    Usage:
        classifier = KNNClassifier(k=3)
        classifier.add_training_data("car", [0.1, 0.5, 0.3, ...])
        classifier.add_training_data("bike", [0.8, 0.2, 0.1, ...])
        
        result = classifier.classify([0.12, 0.48, 0.29, ...])
        # result = ("car", 0.67)  ← label and confidence
    """
    
    def __init__(self, k=3):
        """
        k = how many neighbors to check.
        
        k=3 means "look at the 3 closest training images".
        Odd numbers work best (avoids ties).
        """
        self.k = k
        
        # This will store tuples of (label, features)
        # Example: [("car", [0.1, 0.5, ...]), ("bike", [0.8, 0.2, ...]), ...]
        self.training_data = []
    
    def add_training_data(self, label, features):
        """
        Add one training example.
        
        Args:
            label: what the image is, like "car" or "bike"
            features: list of numbers from feature_extractor
        """
        self.training_data.append((label, features))
    
    def classify(self, new_features):
        """
        Classify a new image.
        
        Args:
            new_features: list of numbers from feature_extractor
            
        Returns:
            tuple: (predicted_label, confidence)
            Example: ("car", 0.67) means "67% sure this is a car"
        """
        if len(self.training_data) == 0:
            return ("unknown", 0.0)
        
        # Step 1: Calculate distance to EVERY training image
        distances = []
        for label, train_features in self.training_data:
            dist = self._euclidean_distance(new_features, train_features)
            distances.append((dist, label))
        
        # Step 2: Sort by distance (closest first)
        distances.sort(key=lambda x: x[0])
        
        # Step 3: Take the K closest neighbors
        k_nearest = distances[:self.k]
        
        # Step 4: Count votes (which label appears most?)
        # Example: k_nearest = [(0.1, "car"), (0.2, "car"), (0.5, "bike")]
        # Votes: {"car": 2, "bike": 1} → Winner: "car"
        labels = [label for _, label in k_nearest]
        vote_counts = Counter(labels)
        
        # The most common label wins!
        winner_label = vote_counts.most_common(1)[0][0]
        winner_count = vote_counts.most_common(1)[0][1]
        
        # Step 5: Calculate confidence
        # confidence = votes_for_winner / total_votes
        # Example: 2 out of 3 = 0.67 = 67%
        confidence = winner_count / len(k_nearest)
        
        return (winner_label, confidence)
    
    def classify_with_details(self, new_features):
        """
        Same as classify(), but returns extra info for the UI.
        
        Returns:
            dict with:
                - 'label': predicted category
                - 'confidence': 0.0 to 1.0
                - 'all_scores': dict of all categories and their scores
                - 'k_nearest': the K closest matches with distances
        """
        if len(self.training_data) == 0:
            return {
                'label': 'unknown',
                'confidence': 0.0,
                'all_scores': {},
                'k_nearest': []
            }
        
        # Calculate all distances
        distances = []
        for label, train_features in self.training_data:
            dist = self._euclidean_distance(new_features, train_features)
            distances.append((dist, label))
        
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # Count all votes
        labels = [label for _, label in k_nearest]
        vote_counts = Counter(labels)
        
        # Calculate scores for ALL categories
        all_categories = set(label for label, _ in self.training_data)
        all_scores = {}
        for cat in all_categories:
            count = vote_counts.get(cat, 0)
            all_scores[cat] = count / len(k_nearest)
            
        # MULTI-OBJECT DETECTION:
        # Instead of picking 1 winner, pick ALL categories that got at least 20% (1 out of 5) votes.
        # This handles images that have "a human AND a bike".
        detected_categories = [cat for cat, score in sorted(all_scores.items(), key=lambda x: -x[1]) if score >= 0.2]
        
        if not detected_categories:
            winner_label = vote_counts.most_common(1)[0][0]
            detected_categories = [winner_label]
            
        # Combine labels (e.g., "human and bike")
        combined_label = " and ".join(detected_categories)
        primary_confidence = all_scores[detected_categories[0]]  # The confidence of the strongest match
        
        # Generate AI Reasoning
        reason = ""
        if len(detected_categories) > 1:
            reason = f"I detected multiple objects! My brain found strong mathematical matches for both {combined_label} mixed together in this image's colors and edges."
        elif primary_confidence == 1.0:
            reason = f"All {self.k} of the closest matching images in my brain are '{detected_categories[0]}'s. The color palette, shape, and edge sharpness are an exact mathematical match for this category."
        elif primary_confidence > 0.5:
            reason = f"The majority ({vote_counts[detected_categories[0]]} out of {len(k_nearest)}) of photos with a similar colored fingerprint and edge density are '{detected_categories[0]}'s."
        else:
            reason = f"This was a trickier one. While the colors resemble a few different things, the closest mathematical match overall falls into the '{detected_categories[0]}' category."
        
        return {
            'label': combined_label,
            'confidence': primary_confidence,
            'reason': reason,
            'all_scores': all_scores,
            'k_nearest': [(dist, label) for dist, label in k_nearest]
        }
    
    def _euclidean_distance(self, features1, features2):
        """
        Calculates the EUCLIDEAN DISTANCE between two feature vectors.
        
        Think of it like measuring the straight-line distance 
        between two points — but in 69 dimensions instead of 2!
        
        Formula: sqrt( (a1-b1)² + (a2-b2)² + ... + (an-bn)² )
        
        Smaller distance = more similar images.
        """
        if len(features1) != len(features2):
            # If features are different lengths, pad the shorter one
            max_len = max(len(features1), len(features2))
            features1 = features1 + [0] * (max_len - len(features1))
            features2 = features2 + [0] * (max_len - len(features2))
        
        # Sum of squared differences
        total = 0
        for a, b in zip(features1, features2):
            total += (a - b) ** 2
        
        return math.sqrt(total)
    
    def get_training_count(self):
        """Returns how many training images have been loaded."""
        return len(self.training_data)
    
    def get_categories(self):
        """Returns a list of all category labels."""
        return list(set(label for label, _ in self.training_data))
