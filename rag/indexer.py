
import os
import json
from pathlib import Path
from rag.retriever import build_faiss_index

# ── Constants ────────────────────────────────────────────────────────────────

KNOWLEDGE_DIR   = Path("rag/knowledge")
CHUNK_SIZE      = 300    # words per chunk
CHUNK_OVERLAP   = 50     # words of overlap between consecutive chunks

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide forward with overlap

    return chunks


# ── Knowledge Base ─────────────────────────────────────────────────────────────

def load_inline_knowledge() -> list[str]:
    knowledge = [
        # ── Transport & Urban ────────────────────────────────────────────
        """Traffic congestion in Indian cities is primarily caused by mixed traffic (two-wheelers, 
        auto-rickshaws, buses, and private cars sharing the same lane), encroachments on roads, 
        poor signal timing (fixed-time vs adaptive), and unplanned urban sprawl. Studies show 
        adaptive signal control systems can reduce average intersection delays by 20-30% with 
        minimal infrastructure investment.""",

        """Bus Rapid Transit (BRT) systems, when properly implemented with dedicated lanes and 
        pre-board ticketing, have achieved journey time reductions of 20-35% in Indian cities 
        like Ahmedabad (Janmarg) and Pune. The failure of Delhi BRT highlights the importance 
        of physical segregation and enforcement over mixed-use corridors.""",

        """Transit-Oriented Development (TOD) concentrates high-density mixed-use development 
        within 500m of mass transit stations, reducing vehicle kilometres travelled by 30-50% 
        compared to sprawl-based development. Mumbai, Bengaluru, and Hyderabad metro projects 
        have adopted TOD policies for station-area planning.""",

        # ── Waste Management ────────────────────────────────────────────
        """India generates approximately 62 million tonnes of solid waste annually, of which 
        only 22-28% is processed. Wet waste (food scraps, garden waste) constitutes 50-60% of 
        total waste in Indian cities. Decentralised composting at ward or apartment-complex level 
        can process 60-70% of wet waste at ₹1,500-3,000/tonne, vs ₹800-1,200/tonne for landfill 
        (not including long-term land and environmental costs).""",

        """Waste pickers (kabadiwalas) recover 15-20% of India's recyclable waste and generate 
        ₹5,000-15,000/month. Formalisation through cooperative models (like SWaCH in Pune or 
        Hasiru Dala in Bengaluru) has improved worker safety, income stability by 40%, and 
        recycling recovery rates without increasing municipal costs.""",

        # ── Water & Environment ──────────────────────────────────────────
        """India's groundwater is being depleted at the rate of 25 km³/year more than recharge. 
        Agriculture accounts for 89% of groundwater extraction. Punjab, Rajasthan, and parts of 
        Gujarat and Maharashtra are critically over-exploited (>100% extraction-to-recharge ratio). 
        Check dams and watershed development programmes under MGNREGA have demonstrated 15-25% 
        improvement in groundwater table levels within 3 years in drought-prone districts.""",

        """Rainwater harvesting in urban areas can offset 25-40% of annual household water demand 
        depending on roof area and rainfall pattern. Bangalore's mandatory rainwater harvesting 
        bylaw (2009) for buildings over 2,400 sq ft has been credited with preventing a 10% 
        additional shortfall in Cauvery water allocation.""",

        """Mangrove forests provide storm surge attenuation of 50-66% per kilometre of mangrove 
        width, significantly reducing coastal flood damage. They also sequester 3-5x more carbon 
        per hectare than tropical forests. India has 4,975 sq km of mangroves (FSI 2021), with 
        the Sundarbans (West Bengal) being the largest mangrove ecosystem in the world.""",

        # ── Public Health ────────────────────────────────────────────────
        """India's doctor-to-population ratio stands at 1:834 (Medical Council of India 2022), 
        but this masks extreme urban-rural inequality: rural areas often have 1:5,000 or worse. 
        Task-shifting — training mid-level health workers (CHOs, paramedics) to diagnose and 
        treat defined conditions using standard protocols — is the WHO-recommended approach to 
        bridge this gap. Pilots in Chhattisgarh and Andhra Pradesh have shown equivalent outcomes 
        for 80% of primary care conditions.""",

        """The Three Delays Model (Thaddeus & Maine, 1994) identifies maternal mortality causes as: 
        Delay 1 — recognising the emergency at household level; Delay 2 — reaching a health facility; 
        Delay 3 — receiving adequate care at the facility. Programs targeting all three delays 
        simultaneously (ASHA training + ambulance + FRU upgrade) have achieved 30-50% MMR reduction 
        in Indian high-MMR districts within 5 years.""",

        """ASER (Annual Status of Education Report) 2022 found that only 42.8% of Grade 5 students 
        in rural India could read a Grade 2 level text. The Teaching at the Right Level (TaRL) 
        methodology — grouping students by learning level rather than grade, and using structured 
        daily reading/numeracy practice — has shown 2-3 grade level improvement in 6-8 months 
        in randomised controlled trials in Bihar, Uttar Pradesh, and Rajasthan.""",

        # ── Agriculture & Rural Economy ──────────────────────────────────
        """India has approximately 146 million farming households, of which 86% are small or 
        marginal (less than 2 hectares). Small farmers face compounded risk: yield risk (weather, 
        pests), price risk (commodity market collapse), and credit risk (moneylender dependency at 
        24-48% annual interest). Farmer Producer Organisations (FPOs) have demonstrated 15-30% 
        higher net income for member farmers through collective bargaining and shared market access.""",

        """The Pradhan Mantri Fasal Bima Yojana (PMFBY) has a claim settlement rate of 42-55% 
        (CAG 2022), far below its potential, due to basis risk (index doesn't reflect actual farm 
        losses), delayed settlement (avg 6-8 months), and compulsory bundling with KCC loans. 
        Demand-based standalone crop insurance (un-bundled from credit) has shown 3x higher 
        voluntary adoption in pilots.""",

        # ── Education & Skilling ─────────────────────────────────────────
        """India's skill development programs (PMKVY, STAR) have trained 13.5 million people 
        (2015-2022) but independent assessments show only 15-20% placement rates in wage employment. 
        Core failures: supply-push design (train first, find jobs later), short certificate courses 
        with no employer trust, poor quality assessment, and no post-placement support. Demand-led 
        skilling programs, where employer demand is assessed first, show 55-70% placement rates.""",

        """India's higher education system has 42,000+ colleges and 1,000+ universities (AISHE 2021), 
        but only 45% of graduates are considered employable in their field of study (NASSCOM-McKinsey 
        2019). Key causes: outdated curricula (syllabi unchanged for 7-10 years), rote-based 
        assessment, no industry partnership, and UGC regulatory constraints on curriculum innovation.""",

        # ── Governance & Welfare ─────────────────────────────────────────
        """India's Public Distribution System (PDS) had a leakage rate of 46.7% (NSSO 2012), 
        which reduced to 28% by 2018 following Aadhaar-linked biometric authentication at fair 
        price shops and end-to-end computerisation. Ghost beneficiaries (fake ration cards) 
        eliminated: 23 crore fake/duplicate ration cards cancelled between 2013-2018. However, 
        Aadhaar authentication failures have caused 3-7% exclusion errors for genuine beneficiaries.""",

        """India's Direct Benefit Transfer (DBT) framework has transferred ₹28 lakh crore 
        (FY2014-2023) to beneficiaries across 315 schemes, saving ₹2.25 lakh crore in leakage 
        reduction. PM-KISAN (₹6,000/year to 11 crore farmers) is the largest DBT scheme by 
        beneficiary count. Aadhaar-linked Jan Dhan accounts are the primary delivery mechanism.""",

        # ── Technology & Digital ─────────────────────────────────────────
        """IndiaStack — the digital public infrastructure comprising Aadhaar (identity), UPI 
        (payments), DigiLocker (documents), and ABDM (health records) — has created a $800 billion+ 
        annual economic value (McKinsey 2019). UPI processed 117 billion transactions worth ₹182 
        lakh crore in FY2023-24, making India the world's largest real-time payments market.""",

        """India's semiconductor import bill stands at $25 billion annually (2022) and is projected 
        to reach $80-100 billion by 2030. The government's India Semiconductor Mission offers 50% 
        capital subsidy for semiconductor fabs and 30% for ATMP (Assembly, Testing, Marking, 
        Packaging) units. Micron Technology's $825 million ATMP facility in Sanand (Gujarat) was 
        the first major commitment under this scheme.""",

        # ── Climate & Energy ─────────────────────────────────────────────
        """India's installed renewable energy capacity reached 190 GW in March 2024 (solar: 82 GW, 
        wind: 44 GW), against a 500 GW target by 2030. Solar power tariffs have fallen from 
        ₹17/kWh (2010) to ₹2.0-2.5/kWh (2024), making solar cheaper than new coal. The key 
        bottleneck for higher renewable penetration is grid flexibility: storage, demand response, 
        and interstate transmission capacity.""",

        """Green hydrogen (GH2) production in India costs $4-6/kg currently (2024), vs a global 
        competitiveness target of $1/kg by 2030. Falling electrolyser costs (20% per year) and 
        ultra-cheap solar power (₹2/kWh) in Rajasthan/Gujarat make India one of the lowest-cost 
        potential producers globally. The National Green Hydrogen Mission targets 5 million 
        tonnes/year of GH2 production by 2030, with 60% earmarked for export.""",

        # ── Mental Health ────────────────────────────────────────────────
        """India has 0.3 psychiatrists per 100,000 population (vs WHO recommendation of 3.0), 
        creating an 80% treatment gap for mental health conditions. The mhGAP (Mental Health Gap 
        Action Programme) — a WHO protocol that trains primary care doctors to diagnose and treat 
        6 common mental health conditions using structured protocols — has been successfully 
        implemented in Rajasthan, Gujarat, and Kerala with 70-80% accuracy compared to specialist 
        diagnosis.""",

        # ── Nutrition ────────────────────────────────────────────────────
        """NFHS-5 (2019-21) found 35.5% of children under 5 are stunted, 19.3% are wasted, 
        and 67.1% of children 6-59 months are anaemic in India. Stunting is driven by the first 
        1,000 days (conception to age 2): inadequate complementary feeding, repeated diarrhoea 
        (WASH link), and maternal anaemia. POSHAN 2.0 integrates ICDS, PM POSHAN (midday meal), 
        and nutrition schemes under a convergence framework.""",
    ]

    return knowledge


def load_file_knowledge(directory: Path) -> list[str]:
    chunks = []

    if not directory.exists():
        return chunks

    for filepath in sorted(directory.iterdir()):
        if filepath.suffix == ".txt":
            text = filepath.read_text(encoding="utf-8")
            file_chunks = chunk_text(text)
            chunks.extend(file_chunks)
            print(f"[indexer] Loaded {len(file_chunks)} chunks from {filepath.name}")

        elif filepath.suffix == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "chunks" in data:
                chunks.extend(data["chunks"])
                print(f"[indexer] Loaded {len(data['chunks'])} chunks from {filepath.name}")

    return chunks


def build_and_save_index() -> None:
    print("[indexer] Loading knowledge base...")

    all_chunks = load_inline_knowledge()
    print(f"[indexer] Inline knowledge: {len(all_chunks)} chunks")

    file_chunks = load_file_knowledge(KNOWLEDGE_DIR)
    all_chunks.extend(file_chunks)
    print(f"[indexer] Total chunks (inline + files): {len(all_chunks)}")

    # Deduplicate (simple exact-match)
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        key = chunk.strip()[:100]  # first 100 chars as key
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    print(f"[indexer] Unique chunks after dedup: {len(unique_chunks)}")

    build_faiss_index(unique_chunks)
    print("[indexer] Index built and saved successfully.")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_and_save_index()
