"""LangGraph workflow builder for ContractLens."""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from contractlens.orchestration.state import (
    ContractLensState,
    create_initial_state,
    get_unverified_spans,
    has_unverified_spans,
    is_complete,
    should_continue_extraction,
    should_continue_retrieval,
    should_continue_verification,
    should_retry_extraction,
)

logger = logging.getLogger(__name__)


def retrieve_chunks_node(state: ContractLensState) -> ContractLensState:
    """Node: Retrieve relevant chunks from contract."""
    logger.info(f"Retrieving chunks for contract {state.contract.contract_id}")
    # Single-document mode: use the full contract text as the context chunk.
    # Full hybrid retrieval (ChromaDB + BM25) is wired in when the vector store is populated.
    if state.contract:
        state.retrieved_chunks = [state.contract.text]
    return state


def extract_clauses_node(state: ContractLensState) -> ContractLensState:
    """Node: Extract clauses for each category."""
    from contractlens.extraction import ClauseExtractor
    
    logger.info(f"Extracting clauses for {len(state.target_categories)} categories")
    
    extractor = ClauseExtractor(default_model=state.model_used)
    
    for category in state.target_categories:
        try:
            spans = extractor.extract(
                contract_text=state.contract.text,
                category=category,
                context="\n".join(state.retrieved_chunks),
                model=state.model_used,
            )
            state.extracted_spans[category] = spans
        except Exception as e:
            logger.error(f"Extraction failed for {category.value}: {e}")
            state.extracted_spans[category] = []
    
    state.extraction_attempts += 1
    return state


def verify_spans_node(state: ContractLensState) -> ContractLensState:
    """Node: Verify extracted spans against source text."""
    from contractlens.verification import SpanVerifier
    
    logger.info("Verifying extracted spans")
    
    verifier = SpanVerifier(default_model=state.model_used)
    unverified = get_unverified_spans(state)
    
    for span, category in unverified:
        try:
            result = verifier.verify(
                span=span,
                source_text=state.contract.text,
                model=state.model_used,
            )
            state.verification_results.append(result)
            
            if result.status.value == "verified":
                # Add to verified clauses
                from contractlens.models import Clause
                import uuid
                
                clause = Clause(
                    clause_id=str(uuid.uuid4()),
                    spans=[span],
                    source_contract_id=state.contract.contract_id,
                )
                state.verified_clauses.append(clause)
            else:
                state.rejected_spans.append(span)
                
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            state.rejected_spans.append(span)
    
    return state


def retry_decision_node(state: ContractLensState) -> str:
    """Node: Decide whether to retry extraction."""
    if should_retry_extraction(state):
        state.retry_count += 1
        state.should_retry = True
        logger.info(f"Retrying extraction (attempt {state.retry_count})")
        return "retry"
    else:
        state.should_retry = False
        return "complete"


def create_extraction_result_node(state: ContractLensState) -> ContractLensState:
    """Node: Create final extraction result."""
    from contractlens.models import ExtractionResult
    import uuid
    
    state.extraction_result = ExtractionResult(
        result_id=str(uuid.uuid4()),
        contract_id=state.contract.contract_id,
        clauses=state.verified_clauses,
        model_used=state.model_used,
        extraction_time_ms=state.total_latency_ms,
        retry_count=state.retry_count,
    )
    
    return state


def build_graph() -> StateGraph:
    """Build the ContractLens LangGraph workflow."""
    
    workflow = StateGraph(ContractLensState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_chunks_node)
    workflow.add_node("extract", extract_clauses_node)
    workflow.add_node("verify", verify_spans_node)
    workflow.add_node("retry_decision", retry_decision_node)
    workflow.add_node("create_result", create_extraction_result_node)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Add edges
    workflow.add_edge("retrieve", "extract")
    workflow.add_edge("extract", "verify")
    
    # Conditional edge from verify
    workflow.add_conditional_edges(
        "verify",
        lambda s: "retry" if should_retry_extraction(s) else "create_result",
        {
            "retry": "retry_decision",
            "create_result": "create_result",
        },
    )
    
    workflow.add_edge("retry_decision", "extract")
    workflow.add_edge("create_result", END)

    return workflow.compile()


def run_workflow(
    contract,
    categories,
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
) -> ContractLensState:
    """Run the complete workflow for a contract."""
    initial_state = create_initial_state(contract, categories, model, max_retries)
    
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    
    return final_state