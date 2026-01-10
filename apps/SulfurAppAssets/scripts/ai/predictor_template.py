import SulfurAI
from typing import Dict, Any, Optional
import datetime
import re
import requests
import json
from dateutil import parser
from google import genai
import textwrap

class Context:
    """Video growth prediction through 4 stages."""

    def __init__(self, channel_page_instance: Any):
        self.channel_page = channel_page_instance
        self.context = {}
        self._channel_data_cache = None

    def stage_1(self, video_id: str, channel_id: str) -> Dict[str, Any]:
        """Get YouTube data + virality score."""
        print("\n[STAGE 1] Collecting basic YouTube data + virality analysis...")

        if not self._channel_data_cache:
            self._channel_data_cache = self.channel_page.fetch_comprehensive_channel_data(
                days_back=90, max_videos=50
            )

        # Find video
        video_data = next((v for v in self._channel_data_cache.get('all_videos', [])
                           if v['video_id'] == video_id), None)
        if not video_data:
            raise ValueError(f"Video {video_id} not found")

        # Get stats
        stats = self.channel_page._load_video_stats_for_ai(video_data, self._channel_data_cache)

        # Build basic data
        views, likes, comments, age_days = (
            stats.get('view_count', 0), stats.get('like_count', 0),
            stats.get('comment_count', 0), stats.get('video_age_days', 1)
        )

        engagement_current = (likes + comments * 2) / max(views, 1)
        engagement_early = engagement_current * (1.2 if age_days <= 7 else 1.5)

        basic_data = {
            "video_id": video_id, "channel_id": channel_id,
            "title": stats.get('title', ''), "description": stats.get('description', ''),
            "views": views, "likes": likes, "comments": comments, "age_days": age_days,
            "engagement_early": engagement_early, "engagement_current": engagement_current,
            "like_rate": stats.get('like_rate', 0), "comment_rate": stats.get('comment_rate', 0),
            "views_per_day": stats.get('views_per_day', 0),
            "duration_seconds": stats.get('video_duration_seconds', 0),
            "is_short": stats.get('is_short', False), "published_at": stats.get('publish_date', ''),
            "channel_subscribers": stats.get('channel_total_subscribers', 0),
            "channel_total_videos": stats.get('channel_total_videos', 0),
            "channel_average_views": stats.get('channel_average_views', 0),
            "relative_performance": stats.get('relative_performance', 0),
        }

        print(f"  ✓ Fetched YouTube data for: {basic_data['title'][:50]}...")
        print(f"  ✓ Views: {views:,} | Age: {age_days} days | Engagement: {engagement_current:.3f}")

        # Fetch historical view data and analyze momentum
        print("  ✓ Fetching historical view data...")
        momentum_data = self._analyze_momentum(video_id, channel_id, age_days)

        print(f"  ✓ Growth momentum: {momentum_data['momentum_factor']:.3f} ({momentum_data['trend_direction']})")

        # Virality analysis
        print("  ✓ Analyzing virality using YouTube-native signals...")
        virality_data = self._analyze_virality(basic_data)

        print(f"  ✓ Virality score: {virality_data['virality_score']:.3f}")
        print(f"  ✓ Growth mode: {virality_data['growth_mode']}")

        self.context.update({
            "basic_data": basic_data,
            "virality": virality_data,
            "momentum": momentum_data
        })
        return {
            "basic_data": basic_data,
            "virality": virality_data,
            "momentum": momentum_data
        }

    def stage_2(self) -> Dict[str, Any]:
        """Generate composite growth score."""
        print("\n[STAGE 2] Generating composite growth score...")

        basic_data = self.context["basic_data"]
        virality = self.context["virality"]

        # Calculate component scores
        scores = {
            "engagement": self._score_engagement(basic_data),
            "virality": virality["virality_score"],
            "authority": self._score_authority(basic_data),
            "velocity": self._score_velocity(basic_data),
            "trend": virality["trend_momentum"]
        }

        # Weighted composite
        weights = {"engagement": 0.25, "virality": 0.30, "authority": 0.20,
                   "velocity": 0.15, "trend": 0.10}
        composite = sum(scores[k] * weights[k] for k in weights)

        result = {
            "composite_growth_score": round(composite, 3),
            "components": {f"{k}_score": round(v, 3) for k, v in scores.items()},
            "weights": weights
        }

        print(f"  ✓ Composite growth score: {composite:.3f}")
        print(f"  ✓ Components: E={scores['engagement']:.2f} V={scores['virality']:.2f} " +
              f"A={scores['authority']:.2f} Vel={scores['velocity']:.2f} T={scores['trend']:.2f}")

        self.context.update(result)
        return result

    def stage_3(self) -> Dict[str, Any]:
        """Calculate growth score, ceiling, baseline with momentum adjustment."""
        print("\n[STAGE 3] Calculating growth score, ceiling, and baseline...")

        basic = self.context["basic_data"]
        composite = self.context["composite_growth_score"]
        momentum = self.context.get("momentum", self._neutral_momentum())

        views, age, vpd, avg_views, subs = (
            basic["views"], basic["age_days"], basic["views_per_day"],
            basic["channel_average_views"], basic["channel_subscribers"]
        )

        # Get momentum data
        momentum_factor = momentum["momentum_factor"]
        trend_direction = momentum["trend_direction"]
        recent_growth = momentum.get("recent_growth_rate", vpd)
        age_category = momentum.get("age_category", "mid")

        # Calculate ceiling
        ceiling = max(subs * 0.3, avg_views * 5) * (1 + composite)

        # Calculate baseline based on momentum
        days_remaining = max(0, 30 - age)

        # Use recent growth rate as the primary signal
        if recent_growth > 0:
            baseline_growth = recent_growth * days_remaining * momentum_factor
        else:
            baseline_growth = vpd * days_remaining * 0.5 * momentum_factor

        baseline = views + baseline_growth

        # Calculate predicted 30d
        growth_potential = (ceiling - baseline) * composite * momentum_factor
        predicted_30d = baseline + growth_potential

        # Constrain to reasonable bounds
        predicted_30d = max(views, min(ceiling, predicted_30d))

        # === ALGORITHMIC PRESSING CHECKS ===

        # Check 1: Dead videos (old + no growth)
        if trend_direction == "dead":
            predicted_30d = views * 1.01  # 1% max growth
            baseline = views * 1.01

        # Check 2: Plateaued videos
        elif trend_direction == "plateaued":
            predicted_30d = min(predicted_30d, views * 1.05)  # 5% max growth
            baseline = min(baseline, views * 1.05)

        # Check 3: Old videos (90+) with low momentum
        elif age > 90:
            if momentum_factor < 0.3:
                predicted_30d = min(predicted_30d, views * 1.1)  # 10% max growth
            elif momentum_factor < 0.5:
                predicted_30d = min(predicted_30d, views * 1.2)  # 20% max growth

        # Check 4: Very old videos (180+) - be extra conservative
        if age > 180 and momentum_factor < 0.5:
            predicted_30d = min(predicted_30d, views * 1.05)

        # Check 5: Young videos with strong absolute growth (don't under-predict)
        if age <= 14 and recent_growth > 10000:
            # Ensure baseline respects strong growth
            min_baseline = views + (recent_growth * days_remaining * 0.5)
            baseline = max(baseline, min_baseline)
            predicted_30d = max(predicted_30d, min_baseline)

        # Determine decay pattern
        virality = self.context["virality"]["virality_score"]

        if trend_direction == "dead":
            pattern = "dead"
        elif trend_direction == "plateaued":
            pattern = "plateaued"
        elif trend_direction in ["massive_growth", "strong_growth"]:
            pattern = "viral_spike"
        elif trend_direction == "accelerating":
            pattern = "accelerating"
        elif trend_direction in ["evergreen_strong", "evergreen_moderate", "evergreen_weak"]:
            pattern = "evergreen"
        elif age_category == "young" and momentum_factor < 0.8:
            pattern = "early_slowdown"
        elif momentum_factor >= 0.9:
            pattern = "steady"
        elif age > 30:
            pattern = "logarithmic"
        else:
            pattern = "slow_burn"

        # Calculate growth score
        headroom = (predicted_30d - baseline) / max(ceiling - baseline, 1)
        base_growth_score = composite * (0.7 + 0.3 * headroom)

        # Apply momentum to growth score
        if trend_direction in ["dead", "plateaued"]:
            growth_score = base_growth_score * 0.1
        elif momentum_factor > 1.0:
            growth_score = base_growth_score * min(1.2, momentum_factor)
        else:
            growth_score = base_growth_score * momentum_factor

        result = {
            "growth_score": round(min(1.0, max(0.0, growth_score)), 3),
            "ceiling": int(ceiling),
            "baseline": int(baseline),
            "predicted_30d": int(predicted_30d),
            "decay_pattern": pattern,
            "momentum_adjusted": True,
            "momentum_factor": momentum_factor,
            "trend_direction": trend_direction,
            "age_category": age_category,
            "confidence_interval": {
                "upper": int(ceiling),
                "lower": int(baseline),
                "expected": int(predicted_30d)
            }
        }

        print(f"  ✓ Growth score: {result['growth_score']:.3f} ({age_category}, {trend_direction})")
        print(f"  ✓ Predicted 30d: {result['predicted_30d']:,} views")
        print(f"  ✓ Range: {result['baseline']:,} - {result['ceiling']:,}")
        print(f"  ✓ Decay pattern: {pattern}")

        self.context.update(result)
        return result

    def stage_4(self, api_key: str) -> Dict[str, Any]:
        """Generate AI explanation using Gemini API with API key.

        Args:
            api_key: Gemini API key from https://aistudio.google.com/app/apikey
        """
        print("\n[STAGE 4] Generating AI explanation with Gemini...")

        # Gather all context data
        context_data = self.get_context()

        # Prepare comprehensive context for AI
        prompt = f"""
        Summarize this YouTube video's performance and growth outlook in one concise paragraph. 
        Include three points: (1) a brief overview of key stats (views: {context_data['basic_data']['views']:,}, age: {context_data['basic_data']['age_days']} days, engagement: {context_data['basic_data']['engagement_current']:.4f}, views/day: {context_data['basic_data']['views_per_day']:,.0f}, virality: {context_data['virality']['virality_score']:.3f}, momentum: {context_data['momentum']['momentum_factor']:.3f}, predicted 30-day views: {context_data['predicted_30d']:,}), (2) why these stats look this way, referencing engagement, virality, momentum, channel size, and recent growth trends, and (3) actionable suggestions for improving future performance, highlighting factors helping or limiting growth. Keep the paragraph compact, clear, and data-driven.

        Video title: {context_data['basic_data']['title']}
        """

        # Use Google AI Studio API with API key
        api_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }

        try:
            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            explanation = response.text

            result = {
                "ai_explanation": explanation,
                "explanation_generated": True
            }

            self.context.update(result)
            return result

        except Exception as e:
            print(f"  ✗ Error calling Gemini SDK: {e}")
            result = {
                "ai_explanation": f"Error generating explanation: {e}",
                "explanation_generated": False
            }
            self.context.update(result)
            return result

    def _discover_or_create_project(self, access_token: str) -> Optional[str]:
        """Try to discover an existing sulfurai-predictor project or create a new one.

        Returns:
            Project ID if successful, None otherwise
        """
        print("  ℹ No project ID provided, checking for sulfurai-predictor projects...")

        try:
            # Try to list existing projects
            project_url = "https://cloudresourcemanager.googleapis.com/v1/projects"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json"
            }

            response = requests.get(project_url, headers=headers, timeout=10)

            if response.status_code == 200:
                projects = response.json().get('projects', [])

                # Look for existing sulfurai-predictor-* projects first
                for project in projects:
                    if (project.get('lifecycleState') == 'ACTIVE' and
                            project['projectId'].startswith('sulfurai-predictor-')):
                        project_id = project['projectId']
                        print(f"  ✓ Found existing sulfurai-predictor project: {project_id}")

                        # Check and enable API if needed
                        if self._check_and_enable_api(access_token, project_id):
                            return project_id

                # No sulfurai-predictor project found, create one
                print("  ℹ No sulfurai-predictor project found, creating one...")
                return self._create_project(access_token)

            elif response.status_code == 403:
                print("  ✗ Permission denied to list projects")
                print("  ℹ This means either:")
                print("     1. Your OAuth token was created BEFORE you added the scope")
                print("        → Solution: Re-authenticate to get a new token")
                print("     2. Cloud Resource Manager API is not enabled")
                print(
                    "        → Enable it at: https://console.cloud.google.com/apis/library/cloudresourcemanager.googleapis.com")
                print("     3. The scope is not properly configured")
                print("        → Required scope: https://www.googleapis.com/auth/cloud-platform")
                print("")
                print("  💡 Quick fix: Just provide your Project ID manually when prompted")
                return None
            else:
                print(f"  ✗ Could not list projects: HTTP {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error discovering projects: {str(e)}")
            return None
        except Exception as e:
            print(f"  ✗ Unexpected error: {str(e)}")
            return None

    def _check_and_enable_api(self, access_token: str, project_id: str) -> bool:
        """Check if Generative Language API is enabled, and enable it if not.

        Returns:
            True if API is enabled or successfully enabled, False otherwise
        """
        try:
            # First check if API is already enabled
            check_url = f"https://serviceusage.googleapis.com/v1/projects/{project_id}/services/generativelanguage.googleapis.com"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
                "x-goog-user-project": project_id
            }

            response = requests.get(check_url, headers=headers, timeout=10)

            if response.status_code == 200:
                service_data = response.json()
                if service_data.get('state') == 'ENABLED':
                    print(f"  ✓ Generative Language API is already enabled")
                    return True

            # API not enabled, try to enable it
            print(f"  ℹ Enabling Generative Language API...")
            enable_url = f"https://serviceusage.googleapis.com/v1/projects/{project_id}/services/generativelanguage.googleapis.com:enable"

            response = requests.post(enable_url, headers=headers, timeout=30)

            if response.status_code in [200, 201]:
                print(f"  ✓ Successfully enabled Generative Language API")
                # Wait a moment for API to be fully enabled
                import time
                time.sleep(2)
                return True
            elif response.status_code == 403:
                print(f"  ⚠ Could not enable API automatically - please enable manually:")
                print(
                    f"     https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project={project_id}")
                return False
            else:
                print(f"  ⚠ API enablement status unclear (HTTP {response.status_code})")
                return False

        except Exception as e:
            print(f"  ⚠ Could not verify API status: {str(e)}")
            return False

    def _create_project(self, access_token: str) -> Optional[str]:
        """Try to create a new Google Cloud project with sulfurai-predictor prefix.

        Returns:
            Project ID if successful, None otherwise
        """
        import random
        import string
        import time

        # Generate unique project ID with sulfurai-predictor prefix
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        project_id = f"sulfurai-predictor-{random_suffix}"

        try:
            create_url = "https://cloudresourcemanager.googleapis.com/v1/projects"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            payload = {
                "projectId": project_id,
                "name": "SulfurAI Video Predictor"
            }

            response = requests.post(create_url, headers=headers, json=payload, timeout=30)

            if response.status_code in [200, 201]:
                print(f"  ✓ Created new project: {project_id}")

                # Wait longer for project to be fully initialized
                print(f"  ⏳ Waiting for project to initialize...")
                time.sleep(5)

                # Enable Service Usage API first (required for enabling other APIs)
                print(f"  ℹ Enabling Service Usage API...")
                service_usage_enabled = self._enable_service_usage_api(access_token, project_id)

                if service_usage_enabled:
                    time.sleep(2)  # Wait for Service Usage API to be ready

                    # Now try to enable Generative Language API
                    if self._check_and_enable_api(access_token, project_id):
                        return project_id

                # If API enablement failed, still return the project ID
                print(f"\n  ⚠ Automatic API setup incomplete")
                print(f"  ℹ Manual setup required:")
                print(
                    f"     1. Go to: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project={project_id}")
                print(f"     2. Click 'Enable' on the Generative Language API")
                print(f"     3. Re-run this script with project ID: {project_id}")
                return project_id

            elif response.status_code == 403:
                print("  ✗ Permission denied to create projects")
                print("  ℹ Your account may not have project creation permissions")
                print("  ℹ Options:")
                print("     1. Ask your Google Cloud admin to create a project for you")
                print("     2. Create a project manually at https://console.cloud.google.com")
                print("     3. Provide an existing project ID when prompted")
                return None
            elif response.status_code == 409:
                print(f"  ⚠ Project ID {project_id} already exists (conflict)")
                print("  ℹ Retrying with a different ID...")
                # Retry once with a new random suffix
                return self._create_project(access_token)
            else:
                print(f"  ✗ Could not create project: HTTP {response.status_code}")
                if response.text:
                    try:
                        error_detail = response.json()
                        print(f"  ℹ Error: {error_detail.get('error', {}).get('message', response.text[:300])}")
                    except:
                        print(f"  ℹ Response: {response.text[:300]}")
                return None

        except Exception as e:
            print(f"  ✗ Error creating project: {str(e)}")
            return None

    def _enable_service_usage_api(self, access_token: str, project_id: str) -> bool:
        """Enable the Service Usage API (required to enable other APIs).

        Returns:
            True if successful, False otherwise
        """
        try:
            enable_url = f"https://serviceusage.googleapis.com/v1/projects/{project_id}/services/serviceusage.googleapis.com:enable"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "x-goog-user-project": project_id
            }

            response = requests.post(enable_url, headers=headers, timeout=30)

            if response.status_code in [200, 201]:
                print(f"  ✓ Service Usage API enabled")
                return True
            else:
                print(f"  ⚠ Could not enable Service Usage API (HTTP {response.status_code})")
                return False

        except Exception as e:
            print(f"  ⚠ Error enabling Service Usage API: {str(e)}")
            return False

    def _check_and_enable_api(self, access_token: str, project_id: str) -> bool:
        """Check if Generative Language API is enabled, and enable it if not.

        Returns:
            True if API is enabled or successfully enabled, False otherwise
        """
        try:
            # First check if API is already enabled
            check_url = f"https://serviceusage.googleapis.com/v1/projects/{project_id}/services/generativelanguage.googleapis.com"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
                "x-goog-user-project": project_id
            }

            response = requests.get(check_url, headers=headers, timeout=10)

            if response.status_code == 200:
                service_data = response.json()
                if service_data.get('state') == 'ENABLED':
                    print(f"  ✓ Generative Language API is already enabled")
                    return True

            # API not enabled, try to enable it
            print(f"  ℹ Enabling Generative Language API...")
            enable_url = f"https://serviceusage.googleapis.com/v1/projects/{project_id}/services/generativelanguage.googleapis.com:enable"

            response = requests.post(enable_url, headers=headers, timeout=30)

            if response.status_code in [200, 201]:
                print(f"  ✓ Successfully enabled Generative Language API")
                # Wait a moment for API to be fully enabled
                import time
                time.sleep(3)
                return True
            elif response.status_code == 403:
                print(f"  ⚠ Permission denied when enabling API")

                # Try to get more details about the error
                try:
                    error_data = response.json()
                    error_message = error_data.get('error', {}).get('message', '')
                    if 'billing' in error_message.lower():
                        print(f"  ℹ This project may need billing enabled:")
                        print(f"     https://console.cloud.google.com/billing/linkedaccount?project={project_id}")
                    else:
                        print(f"  ℹ Error details: {error_message}")
                except:
                    pass

                print(f"  ℹ Please enable the API manually:")
                print(
                    f"     https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project={project_id}")
                return False
            else:
                print(f"  ⚠ API enablement status unclear (HTTP {response.status_code})")
                try:
                    error_data = response.json()
                    print(f"  ℹ Response: {error_data.get('error', {}).get('message', 'Unknown error')}")
                except:
                    pass
                return False

        except Exception as e:
            print(f"  ⚠ Could not verify API status: {str(e)}")
            return False
    def get_context(self) -> Dict[str, Any]:
        """Return complete context."""
        return self.context.copy()

    def reset(self):
        """Reset context."""
        self.context = {}
        self._channel_data_cache = None

    # Internal helper methods
    def _analyze_virality(self, basic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze virality using YouTube-native signals (no external APIs)."""
        # Calculate baseline fit
        baseline_fit = self._calculate_baseline_fit(basic_data)

        # Calculate virality potential
        virality_potential = self._calculate_virality_potential(basic_data)

        # Classify growth mode
        growth_mode = self._classify_growth_mode(baseline_fit, virality_potential)

        # Calculate title hook for additional context
        title_hook = self._score_title(basic_data["title"])

        # Composite virality score (blend of baseline fit + virality potential)
        # Higher weight on virality potential as it captures breakthrough ability
        virality_score = (baseline_fit * 0.35 + virality_potential * 0.50 + title_hook * 0.15)

        # Trend momentum = virality potential (captures acceleration/spike behavior)
        trend_momentum = virality_potential

        return {
            "virality_score": round(virality_score, 3),
            "baseline_fit": baseline_fit,
            "virality_potential": virality_potential,
            "growth_mode": growth_mode,
            "title_hook_score": title_hook,
            "trend_alignment": virality_potential,  # Keep for compatibility
            "niche_saturation": 1.0 - baseline_fit,  # Keep for compatibility
            "trending_keywords": self._extract_keywords(basic_data["title"]),
            "niche_analysis": {
                "trend_alignment": virality_potential,
                "saturation": 1.0 - baseline_fit,
                "is_trending": virality_potential >= 0.7
            },
            "trend_data": {
                "signal_type": "youtube_native_velocity",
                "baseline_fit": baseline_fit,
                "virality_potential": virality_potential,
                "growth_mode": growth_mode
            },
            "trend_momentum": round(trend_momentum, 3)
        }

    def _calculate_baseline_fit(self, basic: Dict[str, Any]) -> float:
        """Calculate how well video performs given channel's normal distribution."""
        vpd = basic["views_per_day"]
        avg_views = max(basic["channel_average_views"], 1)
        expected_vpd = avg_views / 30
        ratio = vpd / max(expected_vpd, 1)

        # Fit is closeness to expectation, not excess
        fit = 1.0 - min(1.0, abs(1.0 - ratio))
        return round(max(0.0, fit), 3)

    def _calculate_virality_potential(self, basic: Dict[str, Any]) -> float:
        """Calculate video's ability to break outside normal distribution."""
        # Acceleration signal
        accel = self._acceleration_signal(basic)

        # Engagement signal
        engagement = self._engagement_signal(basic)

        # Early bonus
        early = 1.0 if basic["age_days"] <= 3 else 0.0

        # Composite (weighted)
        score = accel * 0.5 + engagement * 0.3 + early * 0.2
        return round(min(1.0, score), 3)

    def _acceleration_signal(self, basic: Dict[str, Any]) -> float:
        """Detect if video is accelerating (viral signal)."""
        age = max(basic["age_days"], 1)
        vpd = basic["views_per_day"]
        views = basic["views"]
        expected_linear = views / age
        accel = vpd / max(expected_linear, 1)
        return min(1.0, accel / 2.0)

    def _engagement_signal(self, basic: Dict[str, Any]) -> float:
        """Engagement compression signal (high engagement at scale)."""
        engagement = basic["engagement_current"]
        baseline = 0.05
        return min(1.0, engagement / baseline)

    def _classify_growth_mode(self, baseline_fit: float, virality: float) -> str:
        """Classify growth mode based on baseline fit and virality."""
        if virality >= 0.7:
            return "outlier-driven"
        if baseline_fit >= 0.7:
            return "baseline-consistent"
        return "unstable"

    def _analyze_momentum(self, video_id: str, channel_id: str, age_days: int) -> Dict[str, Any]:
        """Analyze growth momentum using historical daily view data from YouTube Analytics."""
        try:
            # Get access token from channel_page
            if hasattr(self.channel_page, 'access_token') and self.channel_page.access_token:
                access_token = self.channel_page.access_token
            elif hasattr(self.channel_page, 'keyring_name') and self.channel_page.keyring_name:
                from apps.SulfurAppAssets.scripts.essential.ui import tabs_script
                token_data = tabs_script.restore_tokens_from_keyring(self.channel_page.keyring_name)
                provider = token_data.get("provider_token")
                access_token = provider.get("access_token") if isinstance(provider, dict) else provider
            else:
                # Fallback to neutral momentum if no auth available
                return self._neutral_momentum()

            # Fetch daily views from YouTube Analytics API
            headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

            # Calculate date range (from publish date to today)
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=min(age_days, 60))).strftime('%Y-%m-%d')

            params = {
                "ids": f"channel=={channel_id}",
                "startDate": start_date,
                "endDate": end_date,
                "metrics": "views",
                "dimensions": "day",
                "filters": f"video=={video_id}",
                "sort": "day"
            }

            response = requests.get(
                "https://youtubeanalytics.googleapis.com/v2/reports",
                params=params,
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                # Fallback to neutral momentum if API fails
                return self._neutral_momentum()

            data = response.json()
            rows = data.get('rows', [])

            if len(rows) < 3:
                # Not enough data, use neutral momentum
                return self._neutral_momentum()

            # Extract daily view counts
            daily_views = [row[1] for row in rows]  # row format: [date, views]

            # Analyze momentum: compare early period vs recent period
            momentum_result = self._calculate_momentum_factor(daily_views, age_days)

            return momentum_result

        except Exception as e:
            # If anything fails, use neutral momentum (no adjustment)
            print(f"  ⚠ Could not fetch historical data: {e}")
            return self._neutral_momentum()

    def _neutral_momentum(self) -> Dict[str, Any]:
        """Return neutral momentum (no adjustment)."""
        return {
            "momentum_factor": 1.0,
            "trend_direction": "stable",
            "early_growth_rate": 0.0,
            "recent_growth_rate": 0.0,
            "is_plateaued": False,
            "historical_data_available": False
        }

    def _calculate_momentum_factor(self, daily_views: list, age_days: int) -> Dict[str, Any]:
        """Calculate momentum factor with absolute-first logic for young videos."""
        if len(daily_views) < 3:
            return self._neutral_momentum()

        total_days = len(daily_views)

        # Split into periods based on age
        if age_days <= 7:
            mid_point = max(2, total_days // 2)
            early_period = daily_views[:mid_point]
            recent_period = daily_views[mid_point:]
        elif age_days <= 14:
            split = max(2, total_days // 3)
            early_period = daily_views[:split]
            recent_period = daily_views[-split:]
        else:
            split_point = max(3, total_days // 4)
            early_period = daily_views[:split_point]
            recent_period = daily_views[-split_point:]

        early_avg = sum(early_period) / len(early_period) if early_period else 0
        recent_avg = sum(recent_period) / len(recent_period) if recent_period else 0
        peak_views = max(daily_views) if daily_views else 0

        # === YOUNG VIDEOS (≤ 7 days): ABSOLUTE-FIRST LOGIC ===
        if age_days <= 7:
            # For young videos, ABSOLUTE growth matters more than RATIO

            # Tier 1: Massive growth (>100K/day recent)
            if recent_avg >= 100000:
                momentum_factor = 1.2
                trend_direction = "massive_growth"

            # Tier 2: Very strong growth (>50K/day recent)
            elif recent_avg >= 50000:
                momentum_factor = 1.1
                trend_direction = "strong_growth"

            # Tier 3: Strong growth (>10K/day recent)
            elif recent_avg >= 10000:
                momentum_factor = 1.0
                trend_direction = "good_growth"

            # Tier 4: Moderate growth (>1K/day recent)
            elif recent_avg >= 1000:
                # Still growing well, but check ratio
                ratio = recent_avg / early_avg if early_avg > 0 else 1.0
                if ratio >= 0.7:
                    momentum_factor = 0.9
                    trend_direction = "moderate_growth"
                else:
                    momentum_factor = 0.8
                    trend_direction = "slowing_moderate"

            # Tier 5: Weak growth (>100/day recent)
            elif recent_avg >= 100:
                momentum_factor = 0.7
                trend_direction = "weak_growth"

            # Tier 6: Minimal growth (<100/day recent)
            else:
                momentum_factor = 0.5
                trend_direction = "minimal_growth"

        # === MID VIDEOS (7-30 days): BALANCED ABSOLUTE + RATIO ===
        elif age_days <= 30:
            ratio = recent_avg / early_avg if early_avg > 0 else 1.0

            # Strong absolute growth overrides ratio
            if recent_avg >= 5000:
                base_momentum = 1.0
            elif recent_avg >= 1000:
                base_momentum = 0.9
            elif recent_avg >= 500:
                base_momentum = 0.8
            else:
                base_momentum = 0.7

            # Adjust by ratio
            if ratio >= 1.2:
                momentum_factor = min(1.3, base_momentum * 1.2)
                trend_direction = "accelerating"
            elif ratio >= 0.8:
                momentum_factor = base_momentum
                trend_direction = "stable"
            elif ratio >= 0.5:
                momentum_factor = base_momentum * 0.9
                trend_direction = "slowing"
            else:
                momentum_factor = base_momentum * 0.7
                trend_direction = "decelerating"

        # === MATURE VIDEOS (30-90 days): RATIO-FIRST WITH ABSOLUTE CHECKS ===
        elif age_days <= 90:
            ratio = recent_avg / early_avg if early_avg > 0 else 1.0

            # Use ratio as primary signal
            if ratio >= 1.5:
                momentum_factor = 1.2
                trend_direction = "accelerating"
            elif ratio >= 1.1:
                momentum_factor = 1.1
                trend_direction = "growing"
            elif ratio >= 0.9:
                momentum_factor = 1.0
                trend_direction = "stable"
            elif ratio >= 0.6:
                momentum_factor = 0.7
                trend_direction = "slowing"
            elif ratio >= 0.3:
                momentum_factor = 0.5
                trend_direction = "decelerating"
            else:
                momentum_factor = 0.3
                trend_direction = "declining"

            # PRESSING CHECK: Near-zero recent views
            if recent_avg < 50:
                momentum_factor = 0.2
                trend_direction = "plateaued"

        # === OLD VIDEOS (>90 days): STRICT ABSOLUTE THRESHOLDS ===
        else:
            # For old videos, use STRICT absolute thresholds

            if recent_avg >= 1000:
                # Still getting substantial views
                momentum_factor = 0.8
                trend_direction = "evergreen_strong"

            elif recent_avg >= 100:
                # Moderate evergreen traffic
                momentum_factor = 0.5
                trend_direction = "evergreen_moderate"

            elif recent_avg >= 20:
                # Small but consistent trickle
                momentum_factor = 0.2
                trend_direction = "evergreen_trickle"

            else:
                # Essentially dead
                momentum_factor = 0.05
                trend_direction = "dead"

        # === ALGORITHMIC PRESSING CHECKS (override above) ===

        # Check 1: True plateau (recent << peak, old enough)
        if age_days > 14 and peak_views > 5000 and recent_avg < (peak_views * 0.05):
            momentum_factor = min(momentum_factor, 0.25)
            trend_direction = "plateaued"

        # Check 2: Dead old video
        if age_days > 90 and recent_avg < 10:
            momentum_factor = 0.05
            trend_direction = "dead"

        # Check 3: Sustained high growth (don't over-penalize young videos)
        if age_days <= 14 and recent_avg > 5000:
            momentum_factor = max(momentum_factor, 0.85)

        # Check 4: Evergreen verification
        if age_days > 90 and 50 < recent_avg < 500:
            momentum_factor = max(momentum_factor, 0.3)
            if trend_direction == "dead":
                trend_direction = "evergreen_weak"

        return {
            "momentum_factor": round(momentum_factor, 3),
            "trend_direction": trend_direction,
            "early_growth_rate": round(early_avg, 2),
            "recent_growth_rate": round(recent_avg, 2),
            "is_plateaued": trend_direction in ["plateaued", "dead"],
            "historical_data_available": True,
            "daily_view_count": len(daily_views),
            "raw_ratio": round(recent_avg / early_avg, 3) if early_avg > 0 else 1.0,
            "age_category": (
                "young" if age_days <= 7 else
                "mid" if age_days <= 30 else
                "mature" if age_days <= 90 else
                "old"
            ),
            "peak_views": round(peak_views, 2)
        }

    def _extract_keywords(self, title: str) -> list:
        """Extract keywords from title."""
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
                      "for", "of", "with", "by", "from", "is", "are", "was", "were", "be", "been"}
        words = re.findall(r'\b\w+\b', title.lower())
        return [w for w in words if w not in stop_words and len(w) > 2][:10]

    def _score_title(self, title: str) -> float:
        """Score title effectiveness."""
        score = 0.5
        if any(w in title.lower() for w in ["how", "why", "what", "secret", "ultimate"]):
            score += 0.1
        if 40 <= len(title) <= 70:
            score += 0.1
        if any(c in title for c in "!?"):
            score += 0.1
        if title[0].isupper():
            score += 0.05
        if any(word.isupper() for word in title.split()):
            score += 0.05
        if len(title) > 100:
            score -= 0.1
        if title.count("!") > 2:
            score -= 0.1
        return max(0.0, min(1.0, score))

    def _score_engagement(self, basic_data: Dict[str, Any]) -> float:
        """Calculate engagement score."""
        engagement = basic_data["engagement_current"]
        like_rate = basic_data["like_rate"]
        comment_rate = basic_data["comment_rate"]

        eng_norm = min(1.0, engagement / 0.10)
        like_norm = min(1.0, like_rate / 0.05)
        comment_norm = min(1.0, comment_rate / 0.01)

        return min(1.0, eng_norm * 0.5 + like_norm * 0.3 + comment_norm * 0.2)

    def _score_authority(self, basic_data: Dict[str, Any]) -> float:
        """Calculate channel authority score."""
        subs = basic_data["channel_subscribers"]
        avg_views = basic_data["channel_average_views"]

        sub_score = min(1.0, (subs ** 0.5) / 10000)

        if subs > 0:
            view_ratio = avg_views / subs
            ratio_score = min(1.0, view_ratio * 10)
        else:
            ratio_score = 0.0

        return sub_score * 0.6 + ratio_score * 0.4

    def _score_velocity(self, basic_data: Dict[str, Any]) -> float:
        """Calculate growth velocity score."""
        vpd = basic_data["views_per_day"]
        avg = basic_data["channel_average_views"]
        age = basic_data["age_days"]

        if avg > 0:
            expected_daily = avg / 30
            velocity_ratio = vpd / expected_daily if expected_daily > 0 else 0
            velocity_score = min(1.0, velocity_ratio / 2)
        else:
            velocity_score = 0.5

        if age <= 7 and velocity_score > 0.7:
            velocity_score = min(1.0, velocity_score * 1.2)

        return velocity_score


# Mock ChannelPage for testing
class MockChannelPage:
    def __init__(self, keyring_name=None):
        self.keyring_name = keyring_name

    def fetch_comprehensive_channel_data(self, days_back=90, max_videos=50,
                                         max_comments_per_video=5):
        return {
            'fetched_at': datetime.datetime.now().isoformat() + 'Z',
            'all_videos': [{
                'video_id': 'test_video_123',
                'title': 'Ultimate Guide to YouTube Growth in 2025',
                'thumbnail': 'https://example.com/thumb.jpg',
                'view_count': 125000, 'like_count': 8500, 'comment_count': 420,
                'published_at': (datetime.datetime.now() - datetime.timedelta(days=7)).isoformat() + 'Z',
                'duration': 'PT8M45S', 'is_short': False,
                'description': 'Learn proven strategies...'
            }],
            'stats': {
                'total_subscribers': 310000, 'total_videos': 156,
                'total_views': 8500000, 'average_views_per_video': 48000
            }
        }

    def _load_video_stats_for_ai(self, video_data, youtube_data=None):
        publish_date = parser.parse(video_data.get('published_at'))
        age_delta = datetime.datetime.now(datetime.timezone.utc) - publish_date

        views, likes, comments = (
            video_data.get('view_count', 0),
            video_data.get('like_count', 0),
            video_data.get('comment_count', 0)
        )

        return {
            'video_id': video_data.get('video_id'), 'title': video_data.get('title'),
            'description': video_data.get('description', ''),
            'view_count': views, 'like_count': likes, 'comment_count': comments,
            'video_age_days': age_delta.days,
            'like_rate': likes / max(views, 1),
            'comment_rate': comments / max(views, 1),
            'views_per_day': views / max(age_delta.days, 1),
            'video_duration_seconds': 525, 'is_short': False,
            'publish_date': video_data.get('published_at'),
            'channel_total_subscribers': youtube_data.get('stats', {}).get('total_subscribers',
                                                                           0) if youtube_data else 0,
            'channel_total_videos': youtube_data.get('stats', {}).get('total_videos', 0) if youtube_data else 0,
            'channel_average_views': youtube_data.get('stats', {}).get('average_views_per_video',
                                                                       0) if youtube_data else 0,
            'relative_performance': views / youtube_data.get('stats', {}).get('average_views_per_video',
                                                                              1) if youtube_data else 0,
        }


# Standalone function
def get_video_context(
        video_id: str,
        access_token: str,
        gemini_api_key: Optional[str] = None,
        include_ai_explanation: bool = True
) -> Dict[str, Any]:
    """Get video growth context using OAuth. Optionally generate AI explanation with Gemini API.

    Args:
        video_id: The YouTube video ID
        access_token: OAuth access token with YouTube Data API access
        gemini_api_key: Gemini API key from https://aistudio.google.com/app/apikey (optional)
        include_ai_explanation: Whether to generate AI explanation (default: True)
    """

    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    # Step 1: Fetch the specific video directly
    print(f"[1/3] Fetching video {video_id} directly...")
    video_resp = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"part": "snippet,statistics,contentDetails", "id": video_id},
        headers=headers, timeout=30
    )
    video_resp.raise_for_status()
    video_data = video_resp.json()

    if not video_data.get('items'):
        raise ValueError(f"Video {video_id} not found")

    video_item = video_data['items'][0]
    snippet = video_item.get('snippet', {})
    statistics = video_item.get('statistics', {})
    content_details = video_item.get('contentDetails', {})

    # Get channel ID from video
    channel_id = snippet.get('channelId')

    # Check if video belongs to authenticated user's channel
    print(f"[2/3] Verifying video belongs to your channel...")
    channel_resp = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        params={"part": "id", "mine": "true"},
        headers=headers, timeout=30
    )
    channel_resp.raise_for_status()
    my_channel_data = channel_resp.json()

    if not my_channel_data.get('items'):
        raise RuntimeError("Could not fetch your channel")

    my_channel_id = my_channel_data['items'][0]['id']

    if channel_id != my_channel_id:
        raise ValueError(f"Video {video_id} does not belong to your channel (channel ID: {my_channel_id})")

    # Step 2: Fetch channel stats
    print(f"[3/3] Fetching channel statistics...")
    channel_stats_resp = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        params={"part": "statistics", "id": channel_id},
        headers=headers, timeout=30
    )
    channel_stats_resp.raise_for_status()
    channel_stats_data = channel_stats_resp.json()

    if not channel_stats_data.get('items'):
        raise RuntimeError("Could not fetch channel statistics")

    channel_stats = channel_stats_data['items'][0].get('statistics', {})

    # Parse duration
    duration = content_details.get('duration', 'PT0S')
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    duration_seconds = 0
    if match:
        h, m, s = int(match.group(1) or 0), int(match.group(2) or 0), int(match.group(3) or 0)
        duration_seconds = h * 3600 + m * 60 + s

    # Build video data structure
    target_video = {
        'video_id': video_id,
        'title': snippet.get('title', ''),
        'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
        'view_count': int(statistics.get('viewCount', 0)),
        'like_count': int(statistics.get('likeCount', 0)),
        'comment_count': int(statistics.get('commentCount', 0)),
        'published_at': snippet.get('publishedAt', ''),
        'duration': duration,
        'is_short': duration_seconds < 60,
        'description': snippet.get('description', '')[:200]
    }

    # Create channel data dict
    channel_dict = {
        'stats': {
            'total_subscribers': int(channel_stats.get('subscriberCount', 0)),
            'total_videos': int(channel_stats.get('videoCount', 0)),
            'total_views': int(channel_stats.get('viewCount', 0)),
            'average_views_per_video': 0
        },
        'all_videos': [target_video]  # Only contains the target video
    }

    # Calculate average views (approximate from channel total)
    total_videos = channel_dict['stats']['total_videos']
    total_views = channel_dict['stats']['total_views']
    if total_videos > 0:
        channel_dict['stats']['average_views_per_video'] = int(total_views / total_videos)

    print(f"  ✓ Found video: {target_video['title'][:50]}...")
    print(f"  ✓ Channel: {channel_dict['stats']['total_subscribers']:,} subscribers")

    # Create mock channel page
    class _MockChannel:
        def __init__(self, data, access_token=None):
            self._data = data
            self.access_token = access_token
            self.keyring_name = None

        def fetch_comprehensive_channel_data(self, **kwargs):
            return self._data

        def _load_video_stats_for_ai(self, video_data, youtube_data=None):
            publish_date = parser.parse(video_data.get('published_at'))
            age_delta = datetime.datetime.now(datetime.timezone.utc) - publish_date

            views, likes, comments = (
                video_data.get('view_count', 0),
                video_data.get('like_count', 0),
                video_data.get('comment_count', 0)
            )

            match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', video_data.get('duration', 'PT0S'))
            duration_seconds = 0
            if match:
                h, m, s = int(match.group(1) or 0), int(match.group(2) or 0), int(match.group(3) or 0)
                duration_seconds = h * 3600 + m * 60 + s

            stats = {
                'video_id': video_data.get('video_id'), 'title': video_data.get('title'),
                'description': video_data.get('description'),
                'view_count': views, 'like_count': likes, 'comment_count': comments,
                'video_age_days': age_delta.days,
                'like_rate': likes / max(views, 1),
                'comment_rate': comments / max(views, 1),
                'views_per_day': views / max(age_delta.days, 1),
                'video_duration_seconds': duration_seconds,
                'is_short': video_data.get('is_short', False),
                'publish_date': video_data.get('published_at'),
            }

            if youtube_data:
                ch_stats = youtube_data.get('stats', {})
                stats['channel_total_subscribers'] = ch_stats.get('total_subscribers', 0)
                stats['channel_total_videos'] = ch_stats.get('total_videos', 0)
                stats['channel_average_views'] = ch_stats.get('average_views_per_video', 0)
                if stats['channel_average_views'] > 0:
                    stats['relative_performance'] = views / stats['channel_average_views']
                else:
                    stats['relative_performance'] = 0

            return stats

    # Run pipeline
    context = Context(_MockChannel(channel_dict, access_token))
    context.stage_1(video_id, channel_id)
    context.stage_2()
    context.stage_3()

    # Stage 4: AI Explanation (uses Gemini API key)
    if include_ai_explanation and gemini_api_key:
        context.stage_4(gemini_api_key)

    return context.get_context()


if __name__ == "__main__":
    import sys
    from apps.SulfurAppAssets.scripts.essential.ui import tabs_script

    keyring_name = input("Keyring name: ")
    video_id = sys.argv[1] if len(sys.argv) > 1 else input("Video ID: ")

    # Get Gemini API key
    api_key_input = input("Gemini API key (press Enter to skip AI explanation): ").strip()
    api_key = api_key_input if api_key_input else None

    token_data = tabs_script.restore_tokens_from_keyring(keyring_name)
    provider = token_data.get("provider_token")
    access_token = provider.get("access_token") if isinstance(provider, dict) else provider

    try:
        data = get_video_context(video_id, access_token, api_key, include_ai_explanation=bool(api_key))

        print(f"\nTitle: {data['basic_data']['title']}")
        print(f"Growth Score: {data['growth_score']:.3f}")
        print(f"Current Views: {data['basic_data']['views']:,}")
        print(f"Predicted (30d): {data['predicted_30d']:,}")
        print(f"Range: {data['baseline']:,} - {data['ceiling']:,}")
        print(f"Pattern: {data['decay_pattern']}")

        # Print AI explanation if available
        if 'ai_explanation' in data and data.get('explanation_generated', False):
            print("\n" + "=" * 80)
            print("AI EXPLANATION")
            print("=" * 80)
            # Wrap the text to fit a specified width for better readability
            print(textwrap.fill(data['ai_explanation'], width=80))
            print("=" * 80)

    except Exception as e:
        print(f"Error: {e}")