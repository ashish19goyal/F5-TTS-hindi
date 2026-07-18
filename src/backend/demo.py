"""
Comprehensive demo of the integrated TTS pipeline.

This script demonstrates the complete flow:
1. Text Normalization
2. Smart Chunking
3. Inference & Audio Generation

Run with:
    python demo.py
"""

from pathlib import Path

from backend.pipeline import TTSPipeline, print_summary


def demo_simple_sentence():
    """Demo 1: Simple single sentence."""
    print("\n" + "=" * 80)
    print("DEMO 1: Simple Sentence")
    print("=" * 80)

    text = "नमस्ते, दुनिया!"

    with TTSPipeline(max_chunk_chars=500) as pipeline:
        normalized = pipeline.normalizer.normalize(text)
        results = pipeline.process(text)
        print_summary(text, normalized, results)


def demo_multiple_sentences():
    """Demo 2: Multiple sentences."""
    print("\n" + "=" * 80)
    print("DEMO 2: Multiple Sentences")
    print("=" * 80)

    text = """नमस्ते, दुनिया!
    मेरा नाम संजय है।
    मुझे प्रोग्रामिंग पसंद है।"""

    with TTSPipeline(max_chunk_chars=500) as pipeline:
        normalized = pipeline.normalizer.normalize(text)
        results = pipeline.process(text)
        print_summary(text, normalized, results)


def demo_paragraph():
    """Demo 3: Full paragraph."""
    print("\n" + "=" * 80)
    print("DEMO 3: Full Paragraph")
    print("=" * 80)

    text = """
    मेरा नाम राज है। मैं एक सॉफ्टवेयर इंजीनियर हूँ।
    मुझे कंप्यूटर विज्ञान में विशेष रुचि है।
    मैं पिछले पाँच वर्षों से विभिन्न प्रोजेक्ट्स पर काम कर रहा हूँ।
    मेरी पसंदीदा प्रोग्रामिंग भाषा पाइथन है।
     मेरा नाम राज है। मैं एक सॉफ्टवेयर इंजीनियर हूँ।
    मुझे कंप्यूटर विज्ञान में विशेष रुचि है।
    मैं पिछले पाँच वर्षों से विभिन्न प्रोजेक्ट्स पर काम कर रहा हूँ।
    मेरी पसंदीदा प्रोग्रामिंग भाषा पाइथन है।
     मेरा नाम राज है। मैं एक सॉफ्टवेयर इंजीनियर हूँ।
    मुझे कंप्यूटर विज्ञान में विशेष रुचि है।
    मैं पिछले पाँच वर्षों से विभिन्न प्रोजेक्ट्स पर काम कर रहा हूँ।
    मेरी पसंदीदा प्रोग्रामिंग भाषा पाइथन है।
    """

    with TTSPipeline(max_chunk_chars=500) as pipeline:
        normalized = pipeline.normalizer.normalize(text)
        results = pipeline.process(text)
        print_summary(text, normalized, results)


def demo_with_special_characters():
    """Demo 4: Text with special characters and punctuation."""
    print("\n" + "=" * 80)
    print("DEMO 4: Special Characters & Punctuation")
    print("=" * 80)

    text = """क्या आप ठीक हो?
    हाँ, मैं बिल्कुल ठीक हूँ!
    यह अद्भुत है... सच में?
    बिल्कुल सच है!!!  मेरा नाम राज है। मैं एक सॉफ्टवेयर इंजीनियर हूँ।
    मुझे कंप्यूटर विज्ञान में विशेष रुचि है।
    मैं पिछले पाँच वर्षों से विभिन्न प्रोजेक्ट्स पर काम कर रहा हूँ।
    मेरी पसंदीदा प्रोग्रामिंग भाषा पाइथन है।"""

    with TTSPipeline(max_chunk_chars=500) as pipeline:
        normalized = pipeline.normalizer.normalize(text)
        results = pipeline.process(text)
        print_summary(text, normalized, results)


def demo_with_smaller_chunks():
    """Demo 5: Using smaller chunk size."""
    print("\n" + "=" * 80)
    print("DEMO 5: Smaller Chunk Size (max_chars=150)")
    print("=" * 80)

    text = """नमस्ते।
    यह एक विस्तृत उदाहरण है।
    हम अलग-अलग आकार के चंक्स का उपयोग कर रहे हैं।
    प्रत्येक चंक अलग से संसाधित होता है।  मेरा नाम राज है। मैं एक सॉफ्टवेयर इंजीनियर हूँ।
    मुझे कंप्यूटर विज्ञान में विशेष रुचि है।
    मैं पिछले पाँच वर्षों से विभिन्न प्रोजेक्ट्स पर काम कर रहा हूँ।
    मेरी पसंदीदा प्रोग्रामिंग भाषा पाइथन है।  मेरा नाम राज है। मैं एक सॉफ्टवेयर इंजीनियर हूँ।
    मुझे कंप्यूटर विज्ञान में विशेष रुचि है।
    मैं पिछले पाँच वर्षों से विभिन्न प्रोजेक्ट्स पर काम कर रहा हूँ।
    मेरी पसंदीदा प्रोग्रामिंग भाषा पाइथन है।"""

    with TTSPipeline(max_chunk_chars=150) as pipeline:
        normalized = pipeline.normalizer.normalize(text)
        results = pipeline.process(text)
        print_summary(text, normalized, results)


def demo_statistics():
    """Demo 6: Show statistics of processing."""
    print("\n" + "=" * 80)
    print("DEMO 6: Processing Statistics")
    print("=" * 80)

    text = """
    आधुनिक तकनीक ने हमारे जीवन को बदल दिया है।
    कृत्रिम बुद्धिमत्ता और मशीन लर्निंग तेजी से विकसित हो रहे हैं।
    यह प्रौद्योगिकी विभिन्न क्षेत्रों में लागू की जा रही है।
    भाषा प्रसंस्करण एक महत्वपूर्ण अनुप्रयोग है।
     मेरा नाम राज है। मैं एक सॉफ्टवेयर इंजीनियर हूँ।
    मुझे कंप्यूटर विज्ञान में विशेष रुचि है।
    मैं पिछले पाँच वर्षों से विभिन्न प्रोजेक्ट्स पर काम कर रहा हूँ।
    मेरी पसंदीदा प्रोग्रामिंग भाषा पाइथन है।
     मेरा नाम राज है। मैं एक सॉफ्टवेयर इंजीनियर हूँ।
    मुझे कंप्यूटर विज्ञान में विशेष रुचि है।
    मैं पिछले पाँच वर्षों से विभिन्न प्रोजेक्ट्स पर काम कर रहा हूँ।
    मेरी पसंदीदा प्रोग्रामिंग भाषा पाइथन है।
    """

    with TTSPipeline(max_chunk_chars=500) as pipeline:
        # Normalization
        print("\n📝 Step 1: Text Normalization")
        normalized = pipeline.normalizer.normalize(text)
        print(f"   Original length: {len(text)} characters")
        print(f"   Normalized length: {len(normalized)} characters")

        # Chunking
        print("\n📦 Step 2: Text Chunking")
        chunks = pipeline.chunker.chunk(normalized)
        print(f"   Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            print(
                f"     Chunk {i}: {chunk.length} chars - {chunk.text[:50]}..."
            )

        # Inference
        print("\n🔊 Step 3: Inference & Audio Generation")
        results = pipeline.process(text)
        total_latency = sum(r.inference_latency for r in results)
        total_duration = sum(r.duration for r in results)

        print(f"   Total chunks processed: {len(results)}")
        print(f"   Total inference latency: {total_latency*1000:.1f}ms")
        print(f"   Total audio duration: {total_duration:.2f}s")

        # Audio files
        print("\n📁 Generated Audio Files:")
        for result in results:
            file_size = Path(result.audio_path).stat().st_size
            print(
                f"   {Path(result.audio_path).name}: {file_size} bytes"
            )


def main():
    """Run all demos."""
    print("\n" + "🎯 " * 20)
    print("HINDI TEXT-TO-SPEECH PIPELINE DEMO")
    print("Demonstrating: Normalization → Chunking → Inference")
    print("🎯 " * 20)

    try:
        demo_simple_sentence()
        demo_multiple_sentences()
        demo_paragraph()
        demo_with_special_characters()
        demo_with_smaller_chunks()
        demo_statistics()

        print("\n" + "=" * 80)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
