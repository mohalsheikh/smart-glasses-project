from __future__ import annotations

from typing import Optional, List, Dict, Any


class VisionHandlersMixin:

    def _handle_vision_qa(self, frame, detections, text: str) -> str:
        """Answer arbitrary questions about the current visual scene.

        Examples:
          - “Is the stove on or off?”
          - “Is this door open?”
          - “Which button should I press?”
          - “Is there anything in front of me?”
        """
        question = (text or "").strip()
        if not question:
            return "What would you like me to check?"

        # Safety-first instruction: if uncertain, ask for a better view instead of guessing.
        instruction = (
            "Answer the user's question using only what is visible right now. "
            "Be safety-first. If you are not confident, say you are unsure and ask the user to move closer, "
            "change angle, or point the camera at the specific control/label. "
            "When possible, include a one-sentence reason for your answer. "
            "Keep it concise.\n\n"
            f"User question: {question}"
        )

        fallback = (
            "I can't tell for sure. Try moving a bit closer and aim at the specific control, label, or indicator."
        )

        try:
            return self.scene_ai.describe_scene(
                frame=frame,
                detections=list(detections or []),
                question=instruction,
                fallback_text=fallback,
                mode="qa",
            )
        except Exception:
            return fallback

    def _handle_describe_env(self, frame, detections: Optional[List[Dict[str, Any]]] = None) -> str:
        if frame is None:
            return "I don't have a clear view right now. Make sure nothing is blocking the camera."

        detections = detections or []
        fallback = "I can see your surroundings but I'm having trouble describing them clearly."

        if self.quick_mode:
            question = (
                "Briefly describe the user's surroundings in 1–2 short sentences. "
                "Mention only the most important things: setting type, main objects, "
                "and whether there are people nearby. Speak in second person."
            )
        else:
            question = (
                "Describe everything visible from this viewpoint in a natural, conversational way. "
                "Include: the setting/location type, main objects and their arrangement, any people "
                "(describe generically like 'a person' or 'two people'), visible text on signs or labels, "
                "lighting and atmosphere, and anything notable or interesting. "
                "Speak in second person as if talking to the wearer. Be detailed but concise."
            )

        try:
            return self.scene_ai.describe_scene(
                frame=frame,
                detections=detections,
                question=question,
                fallback_text=fallback,
                mode="narration",
            )
        except Exception as e:
            print(f"❌ Error in _handle_describe_env: {e!r}")
            return fallback

    def _handle_describe_person(self, frame, detections: Optional[List[Dict[str, Any]]] = None) -> str:
        detections = detections or []
        num_people = self._count_people_from_detections(detections)

        if frame is None:
            if num_people == 0:
                return "I don't see anyone in front of you right now."
            if num_people == 1:
                return "I see one person in front of you, but I need a clearer view to describe them."
            return f"I see {num_people} people in front of you."

        if self.quick_mode:
            base_fallback = "I see a person in front of you."
            question = (
                "In 1–2 short sentences, describe the main person in view focusing on their clothing "
                "and what they're doing. Do NOT describe facial features, body shape, estimated age, "
                "gender, ethnicity, or identity. Speak in second person."
            )
        else:
            base_fallback = "I see a person but can't describe them clearly at the moment."
            question = (
                "Describe the main person in view focusing on: their clothing style and colors, "
                "what they're doing or their posture, anything they're holding or wearing "
                "(bags, accessories), and their general positioning in the scene. "
                "DO NOT describe: facial features, body shape, estimated age, gender, ethnicity, "
                "or attempt to identify them. Keep it respectful and focused on observable, "
                "non-sensitive details. Speak in second person."
            )

        try:
            answer = self.scene_ai.describe_scene(
                frame=frame,
                detections=detections,
                question=question,
                fallback_text=base_fallback,
                mode="narration",
            )

            lowered = (answer or "").lower()
            if any(
                phrase in lowered
                for phrase in [
                    "i'm sorry, i can't",
                    "i cannot",
                    "i'm unable to",
                    "i can't help with that",
                    "i shouldn't",
                ]
            ):
                if num_people == 0:
                    return "I don't see anyone clearly in front of you."
                if num_people == 1:
                    return "I see one person in front of you."
                return f"I see {num_people} people in front of you."

            return (answer or "").strip() or base_fallback

        except Exception as e:
            print(f"❌ Error in _handle_describe_person: {e!r}")
            if num_people == 1:
                return "I see one person in front of you."
            if num_people > 1:
                return f"I see {num_people} people in front of you."
            return "I don't see anyone clearly right now."

    def _handle_people_presence(self, detections: Optional[List[Dict[str, Any]]] = None) -> str:
        num_people = self._count_people_from_detections(detections)
        if num_people == 0:
            return "I don't see any people around you at the moment."
        if num_people == 1:
            return "Yes, I see one person around you."
        return f"I see {num_people} people around you."

    def _handle_scene_change(self, detections: Optional[List[Dict[str, Any]]] = None) -> str:
        current = detections or self.last_scene_detections
        previous = self.prev_scene_detections

        if not current or not previous:
            return "I don't have enough recent information yet to tell what changed."

        dt = self.last_scene_time - self.prev_scene_time
        print(f"⏱  Scene change delta time: {dt:.2f}s")

        people_prev = self._count_people_from_detections(previous)
        people_now = self._count_people_from_detections(current)

        def label_set(dets: List[Dict[str, Any]]) -> set:
            s = set()
            for d in dets:
                label = (d.get("label") or "").lower().strip()
                if label:
                    s.add(label)
            return s

        labels_prev = label_set(previous)
        labels_now = label_set(current)

        ignore_people_labels = {
            "human face",
            "human body",
            "person",
            "man",
            "woman",
            "boy",
            "girl",
            "people",
        }

        objs_prev = {l for l in labels_prev if l not in ignore_people_labels}
        objs_now = {l for l in labels_now if l not in ignore_people_labels}

        new_objs = objs_now - objs_prev
        gone_objs = objs_prev - objs_now

        parts: List[str] = []

        if people_now > people_prev:
            diff = people_now - people_prev
            parts.append("someone just came into view" if diff == 1 else f"{diff} people just came into view")
        elif people_now < people_prev:
            diff = people_prev - people_now
            parts.append(
                "someone just left your surroundings" if diff == 1 else f"{diff} people just left your surroundings"
            )

        if new_objs:
            parts.append(f"you now have {', '.join(sorted(new_objs))} in view")
        if gone_objs:
            parts.append(f"you no longer have {', '.join(sorted(gone_objs))} in view")

        if not parts:
            return "Nothing major seems to have changed in the last few seconds."

        return f"In the last few seconds, {', and '.join(parts)}."

    def _handle_read_text(self, frame, detections: Optional[List[Dict[str, Any]]] = None) -> str:
        if frame is None:
            return "I don't have a clear view to read text right now."

        detections = detections or []
        fallback = "I can't make out any readable text clearly."

        try:
            return self.scene_ai.describe_scene(
                frame=frame,
                detections=detections,
                question=(
                    "Read all clearly visible text in this image out loud. Include text from: signs, "
                    "labels, screens, documents, posters, packaging, or any printed/digital text. "
                    "Read exactly what you see, in a natural order. If no text "
                    "is clearly readable, say so."
                ),
                fallback_text=fallback,
                mode="ocr",
            )
        except Exception as e:
            print(f"❌ Error in _handle_read_text: {e!r}")
            return fallback

    def _handle_answer_visible_question(
        self,
        frame,
        detections: Optional[List[Dict[str, Any]]] = None,
        original_query: str = "",
    ) -> str:
        if frame is None:
            return (
                "I need to see the question to solve it. "
                "Try holding the page or screen steady in front of the camera."
            )

        detections = detections or []

        if self.quick_mode:
            question = (
                "You are helping a visually impaired user wearing smart glasses.\n"
                "They are showing you a written question or math problem on paper or a screen.\n\n"
                "Your job:\n"
                "1) Read the main question in one short sentence starting with 'Question:'.\n"
                "2) Then say 'Answer:' followed by the final answer only.\n\n"
                "VOICE RULES:\n"
                "- Use ONLY plain text, no LaTeX.\n"
                "- Fractions like '7/8' or '7 over 8'.\n"
                "- Keep the whole response under about 40 words.\n"
                "- Do not talk about the image/camera or being an AI."
            )
        else:
            question = (
                "You are helping a visually impaired user wearing smart glasses.\n"
                "They are showing you a written question or math problem on paper or a screen.\n\n"
                "Your job:\n"
                "1) Read the main question in ONE short sentence starting with 'Question:'.\n"
                "2) Then give a few short steps (at most 3–4), each starting with 'Step 1:', 'Step 2:', etc.\n"
                "3) Finally, clearly say 'Answer:' followed by the final answer.\n\n"
                "VOICE RULES:\n"
                "- Spoken-friendly. No LaTeX.\n"
                "- For fractions, use '7 over 8' or '7/8'.\n"
                "- Keep it under ~80–100 words.\n"
                "- Do not mention the image/camera or being an AI."
            )

        fallback = (
            "I can see the question, but I'm having trouble solving it clearly. "
            "You might need to zoom in or adjust the lighting."
        )

        try:
            answer = self.scene_ai.describe_scene(
                frame=frame,
                detections=detections,
                question=question,
                fallback_text=fallback,
                mode="ocr",
            )

            answer = (answer or "").strip()
            if "answer" not in answer.lower():
                answer = answer + "\nAnswer: that's the final result."

            return answer
        except Exception as e:
            print(f"❌ Error in _handle_answer_visible_question: {e!r}")
            return fallback

    def _handle_translate_visible_text(
        self,
        frame,
        detections: Optional[List[Dict[str, Any]]] = None,
        original_query: str = "",
        target_language_param: Optional[str] = None,
    ) -> str:
        if frame is None:
            return (
                "I need to see the text to translate it. "
                "Try holding the page or screen steady in front of the camera."
            )

        detections = detections or []
        target_language = target_language_param or self._parse_target_language(original_query)
        pretty_lang = target_language.capitalize()

        if self.quick_mode:
            question = (
                "You are helping a visually impaired user wearing smart glasses.\n"
                "They are showing you some written text on paper or a screen.\n\n"
                f"Your job:\n"
                f"1) Read the main visible text very briefly.\n"
                f"2) Then give the translation into {pretty_lang}.\n\n"
                "Format:\n"
                "- 'Original:' (short)\n"
                f"- 'Translation ({pretty_lang}):'\n\n"
                "VOICE RULES:\n"
                "- Plain text only.\n"
                "- Do not mention the image/camera or being an AI."
            )
        else:
            question = (
                "You are helping a visually impaired user wearing smart glasses.\n"
                "They are showing you written text on paper or a screen.\n\n"
                "Your job:\n"
                "1) Read the clearly visible text in its original language.\n"
                f"2) Then provide a natural translation into {pretty_lang}.\n\n"
                "Format:\n"
                "- First line: 'Original:'\n"
                f"- Second line: 'Translation ({pretty_lang}):'\n\n"
                "VOICE RULES:\n"
                "- Plain text only.\n"
                "- Do not mention the image/camera or being an AI."
            )

        fallback = (
            "I'm having trouble reading the text clearly. "
            "Try moving the camera closer, holding it steady, or improving the lighting."
        )

        try:
            answer = self.scene_ai.describe_scene(
                frame=frame,
                detections=detections,
                question=question,
                fallback_text=fallback,
                mode="ocr",
            )
            return (answer or "").strip() or fallback
        except Exception as e:
            print(f"❌ Error in _handle_translate_visible_text: {e!r}")
            return fallback

    def _handle_identify_object(
        self,
        frame,
        detections: Optional[List[Dict[str, Any]]] = None,
        original_query: str = "",
    ) -> str:
        if frame is None:
            return "I need to see what you're looking at. Make sure the camera has a clear view."

        detections = detections or []
        fallback = "I can see something but I'm having trouble identifying it clearly."

        try:
            return self.scene_ai.describe_scene(
                frame=frame,
                detections=detections,
                question=(
                    f"The user asked: '{original_query}'. Identify and describe the main object(s) "
                    "they're looking at. Be specific about what it is, its key features, color, "
                    "condition, and any relevant details. If there are multiple objects, focus on "
                    "the most prominent or central one unless asked otherwise."
                ),
                fallback_text=fallback,
                mode="narration",
            )
        except Exception as e:
            print(f"❌ Error in _handle_identify_object: {e!r}")
            return fallback

    def _handle_find_object(
        self,
        frame,
        detections: Optional[List[Dict[str, Any]]] = None,
        object_name: Optional[str] = None,
        original_query: str = "",
    ) -> str:
        if not object_name:
            object_name = "that object"

        if frame is None:
            return f"I need to see the area to help you find {object_name}."

        detections = detections or []
        fallback = f"I'm looking for {object_name} but I'm not sure if I can see it clearly."

        try:
            return self.scene_ai.describe_scene(
                frame=frame,
                detections=detections,
                question=(
                    f"The user is looking for: {object_name}. Look carefully at the image. "
                    f"If you can see {object_name}, describe its location (left, right, center, "
                    "top, bottom) and what's around it. If you see multiple similar objects, "
                    "describe each location. If you cannot see it, say so clearly and mention "
                    "what you CAN see that might be relevant."
                ),
                fallback_text=fallback,
                mode="narration",
            )
        except Exception as e:
            print(f"❌ Error in _handle_find_object: {e!r}")
            return fallback

    def _handle_appearance_opinion(
        self,
        frame,
        detections: Optional[List[Dict[str, Any]]],
        text: str,
    ) -> str:
        detections = detections or []
        base_desc: Optional[str] = None

        try:
            base_desc = self._handle_describe_person(frame, detections)
        except Exception as e:
            print(f"⚠️ Error getting base person description for appearance_opinion: {e!r}")

        if base_desc:
            return (
                "I can't really judge whether someone looks 'good' or 'pretty', "
                "but I can tell you how you appear from what I can see. "
                + base_desc
            )

        return "I can't judge attractiveness, but from what I can see you look put together."
