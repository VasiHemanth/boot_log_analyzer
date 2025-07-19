import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.llms import Ollama
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field

# Enhanced data models
class LogAnalysisResult(BaseModel):
    phase: str = Field(description="Boot phase (SEC, PEIM_LOADING, DXE_HANDOFF, etc.)")
    status: str = Field(description="Status (SUCCESS, WARNING, ERROR)")
    component: str = Field(description="Component name")
    issues_found: List[str] = Field(description="List of issues or anomalies detected")
    performance_metrics: Dict[str, Any] = Field(description="Performance related metrics")
    memory_usage: Dict[str, str] = Field(description="Memory allocation information")
    recommendations: List[str] = Field(description="Suggested actions or recommendations")

class BatchAnalysisResult(BaseModel):
    total_entries: int
    error_count: int
    warning_count: int
    critical_issues: List[str]
    performance_summary: Dict[str, Any]
    overall_status: str

@dataclass
class ExtractedLogData:
    timestamp: str
    component: str
    action: str
    address: Optional[str]
    guid: Optional[str]
    details: str
    raw_line: str
    phase: str
    status: str


class EDKIILogParser:
    def __init__(self):
        self.patterns = {
            # Boot phases
            'sec_start': re.compile(r'SEC Has Started'),
            'peim_loading': re.compile(r'Loading PEIM\s+([0-9A-Fa-f\-]+)'),
            'peim_entry': re.compile(r'Loading PEIM at\s+(0x[0-9A-Fa-f]+)\s+EntryPoint\s*=\s*(0x[0-9A-Fa-f]+)\s+(\w+\.efi)'),
            'dxe_loading': re.compile(r'Loading DXE CORE at\s+(0x[0-9A-Fa-f]+)\s+EntryPoint\s*=\s*(0x[0-9A-Fa-f]+)'),
            'driver_loading': re.compile(r'Loading driver\s+([0-9A-Fa-f\-]+)'),
            'driver_entry': re.compile(r'Loading driver at\s+(0x[0-9A-Fa-f]+)\s+EntryPoint\s*=\s*(0x[0-9A-Fa-f]+)\s+(\w+\.efi)'),
            
            # Memory operations
            'memory_alloc': re.compile(r'Memory Allocation\s+(0x[0-9A-Fa-f]+)\s+(0x[0-9A-Fa-f]+)\s*-\s*(0x[0-9A-Fa-f]+)'),
            'hoblist': re.compile(r'HOBLIST address.*?=\s*(0x[0-9A-Fa-f]+)'),
            'stack_info': re.compile(r'Stack Base:\s*(0x[0-9A-Fa-f]+),\s*Stack Size:\s*(0x[0-9A-Fa-f]+)'),
            'memory_test': re.compile(r'(\d+)\s+bytes of system memory tested'),
            
            # Protocol operations
            'install_protocol': re.compile(r'InstallProtocolInterface:\s+([0-9A-Fa-f\-]+)\s+([0-9A-Fa-f]+)'),
            'install_ppi': re.compile(r'Install PPI:\s+([0-9A-Fa-f\-]+)'),
            'notify_ppi': re.compile(r'Notify:\s+PPI Guid:\s+([0-9A-Fa-f\-]+).*?entry point:\s+(0x[0-9A-Fa-f]+)'),
            
            # Library operations
            'load_library': re.compile(r'LoadLibraryEx\s+\(\s*([^,]+),.*?\)\s+@\s+(0x[0-9A-Fa-f]+)'),
            
            # Error patterns
            'error': re.compile(r'(Error|Failed|Exception|Invalid)', re.IGNORECASE),
            'warning': re.compile(r'(Warning|Warn)', re.IGNORECASE),
            
            # Boot options and BDS
            'bds_entry': re.compile(r'\[Bds\]\s+Entry'),
            'boot_option': re.compile(r'Boot\d+:\s+(.+?)\s+0x[0-9A-Fa-f]+'),
            'booting': re.compile(r'\[Bds\]Booting\s+(.+)'),
            
            # Console and graphics
            'graphics_console': re.compile(r'GraphicsConsole.*?resolution\s+(\d+)\s+x\s+(\d+)'),
            'console_mode': re.compile(r'Graphics - Mode\s+(\d+),\s+Column\s*=\s*(\d+),\s+Row\s*=\s*(\d+)'),
            
            # General patterns
            'progress_code': re.compile(r'PROGRESS CODE:\s+([V0-9A-Fa-f]+)\s+([I0-9A-Fa-f]+)'),
            'address': re.compile(r'0x[0-9A-Fa-f]+'),
            'guid': re.compile(r'[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}')
        }
    
    def parse_log_file(self, log_content: str) -> List[ExtractedLogData]:
        """Parse the complete log file and extract structured data"""
        entries = []
        lines = log_content.strip().split('\n')
        
        for i, line in enumerate(lines):
            entry = self._parse_line(line, i)
            if entry:
                entries.append(entry)
        
        return entries
    
    def _parse_line(self, line: str, line_number: int) -> Optional[ExtractedLogData]:
        """Parse individual log line"""
        line = line.strip()
        if not line or line.startswith('FreeLibrary'):
            return None
        
        timestamp = f"line_{line_number:06d}"
        component = "UNKNOWN"
        action = "UNKNOWN"
        address = None
        guid = None
        phase = "UNKNOWN"
        status = "INFO"
        
        # Extract GUIDs and addresses
        guid_matches = self.patterns['guid'].findall(line)
        guid = guid_matches[0] if guid_matches else None
        
        address_matches = self.patterns['address'].findall(line)
        address = address_matches[0] if address_matches else None
        
        # Determine status
        if self.patterns['error'].search(line):
            status = "ERROR"
        elif self.patterns['warning'].search(line):
            status = "WARNING"
        elif any(keyword in line.lower() for keyword in ['loading', 'install', 'entry', 'start']):
            status = "SUCCESS"
        
        # Parse specific patterns
        if self.patterns['sec_start'].search(line):
            phase = "SEC_INITIALIZATION"
            component = "SEC_CORE"
            action = "START"
        
        elif match := self.patterns['peim_entry'].search(line):
            phase = "PEIM_LOADING"
            component = match.group(3)
            action = "LOAD_PEIM"
            address = match.group(1)
        
        elif match := self.patterns['dxe_loading'].search(line):
            phase = "DXE_HANDOFF"
            component = "DXE_CORE"
            action = "LOAD_DXE"
            address = match.group(1)
        
        elif match := self.patterns['driver_entry'].search(line):
            phase = "DXE_DRIVER_LOADING"
            component = match.group(3)
            action = "LOAD_DRIVER"
            address = match.group(1)
        
        elif match := self.patterns['memory_alloc'].search(line):
            phase = "MEMORY_MANAGEMENT"
            component = "MEMORY_ALLOCATOR"
            action = "ALLOCATE_MEMORY"
        
        elif match := self.patterns['install_protocol'].search(line):
            phase = "PROTOCOL_INSTALLATION"
            component = "PROTOCOL_MANAGER"
            action = "INSTALL_PROTOCOL"
            guid = match.group(1)
        
        elif match := self.patterns['install_ppi'].search(line):
            phase = "PPI_INSTALLATION"
            component = "PPI_MANAGER"
            action = "INSTALL_PPI"
            guid = match.group(1)
        
        elif match := self.patterns['bds_entry'].search(line):
            phase = "BDS_PHASE"
            component = "BDS_CORE"
            action = "BDS_ENTRY"
        
        elif match := self.patterns['graphics_console'].search(line):
            phase = "CONSOLE_INITIALIZATION"
            component = "GRAPHICS_CONSOLE"
            action = "SET_RESOLUTION"
        
        return ExtractedLogData(
            timestamp=timestamp,
            component=component,
            action=action,
            address=address,
            guid=guid,
            details=line,
            raw_line=line,
            phase=phase,
            status=status
        )


class EDKIIPromptTemplates:
    def __init__(self):
        # System message for context
        self.system_message = """
        You are an expert UEFI/EDK II firmware boot process analyzer. You understand:
        - SEC (Security) Phase initialization
        - PEI (Pre-EFI Initialization) Phase with PEIMs
        - DXE (Driver Execution Environment) Phase
        - BDS (Boot Device Selection) Phase
        - Memory management and HOB (Hand-Off Blocks)
        - PPI (PEIM-to-PEIM Interface) and Protocol installations
        - Common boot issues and performance bottlenecks
        
        Analyze the provided boot log data and provide structured insights.
        """
        
        # Individual log entry analysis
        self.entry_analysis_template = PromptTemplate(
            input_variables=["log_entry", "context"],
            template="""
            Analyze this EDK II boot log entry:
            
            Entry: {log_entry}
            Context: {context}
            
            Provide analysis in the following format:
            - Phase: [SEC_INITIALIZATION|PEIM_LOADING|DXE_HANDOFF|DXE_DRIVER_LOADING|BDS_PHASE|etc.]
            - Status: [SUCCESS|WARNING|ERROR]
            - Component: [component name]
            - Issues Found: [list any issues or anomalies]
            - Performance Metrics: [any relevant performance data]
            - Memory Usage: [memory allocation information if present]
            - Recommendations: [suggested actions if any]
            
            Focus on identifying:
            1. Memory allocation patterns and potential leaks
            2. Driver loading failures or delays
            3. Protocol/PPI installation issues
            4. Performance bottlenecks
            5. Security concerns
            """
        )
        
        # Batch log analysis
        self.batch_analysis_template = PromptTemplate(
            input_variables=["log_entries", "error_count", "total_entries"],
            template="""
            Analyze this batch of EDK II boot log entries:
            
            Total Entries: {total_entries}
            Error Count: {error_count}
            
            Log Entries:
            {log_entries}
            
            Provide comprehensive analysis:
            
            1. OVERALL BOOT STATUS: [SUCCESS|WARNING|CRITICAL]
            
            2. PHASE ANALYSIS:
            - SEC Phase: [status and issues]
            - PEI Phase: [PEIM loading status, memory initialization]
            - DXE Phase: [driver loading, protocol installation]
            - BDS Phase: [boot device selection, console setup]
            
            3. CRITICAL ISSUES FOUND:
            - List any boot-blocking errors
            - Memory allocation failures
            - Driver load failures
            - Security vulnerabilities
            
            4. PERFORMANCE METRICS:
            - Boot time analysis
            - Memory usage efficiency
            - Driver loading times
            
            5. RECOMMENDATIONS:
            - Priority fixes needed
            - Performance optimizations
            - Security improvements
            
            Format response as structured analysis suitable for automated processing.
            """
        )
        
        # Error-focused analysis
        self.error_analysis_template = PromptTemplate(
            input_variables=["error_entries", "surrounding_context"],
            template="""
            Analyze these EDK II boot errors in detail:
            
            Error Entries:
            {error_entries}
            
            Surrounding Context:
            {surrounding_context}
            
            For each error, provide:
            1. ERROR TYPE: [Driver Load Failure|Memory Error|Protocol Issue|etc.]
            2. ROOT CAUSE: [detailed analysis of why this occurred]
            3. IMPACT: [boot impact - blocking/non-blocking]
            4. RESOLUTION: [specific steps to fix]
            5. PREVENTION: [how to prevent in future]
            
            Prioritize errors by severity and boot impact.
            """
        )
        
        # Performance analysis
        self.performance_analysis_template = PromptTemplate(
            input_variables=["memory_entries", "timing_data", "driver_load_data"],
            template="""
            Analyze EDK II boot performance:
            
            Memory Operations:
            {memory_entries}
            
            Timing Data:
            {timing_data}
            
            Driver Loading:
            {driver_load_data}
            
            Provide performance analysis:
            
            1. MEMORY EFFICIENCY:
            - Total memory allocated
            - Allocation patterns
            - Potential fragmentation
            - Memory leaks
            
            2. BOOT TIME ANALYSIS:
            - Phase durations
            - Slowest components
            - Bottlenecks identified
            
            3. DRIVER LOADING PERFORMANCE:
            - Load times per driver
            - Failed loads and impact
            - Loading sequence efficiency
            
            4. OPTIMIZATION RECOMMENDATIONS:
            - Memory optimization opportunities
            - Boot time improvements
            - Driver loading optimizations
            """
        )


class EDKIIAnalyzerWithLangChain:
    def __init__(self, model_name: str = "llama2", temperature: float = 0.1):
        self.parser = EDKIILogParser()
        self.templates = EDKIIPromptTemplates()
        self.llm = Ollama(model=model_name, temperature=temperature)
        
        # Setup output parsers
        self.analysis_parser = PydanticOutputParser(pydantic_object=LogAnalysisResult)
        self.batch_parser = PydanticOutputParser(pydantic_object=BatchAnalysisResult)
        
        # Create chains
        self.setup_chains()
    
    def setup_chains(self):
        """Setup LangChain chains for different analysis types"""
        from langchain.chains import LLMChain
        
        # Individual entry analysis chain
        self.entry_chain = LLMChain(
            llm=self.llm,
            prompt=self.templates.entry_analysis_template,
            verbose=True
        )
        
        # Batch analysis chain
        self.batch_chain = LLMChain(
            llm=self.llm,
            prompt=self.templates.batch_analysis_template,
            verbose=True
        )
        
        # Error analysis chain
        self.error_chain = LLMChain(
            llm=self.llm,
            prompt=self.templates.error_analysis_template,
            verbose=True
        )
        
        # Performance analysis chain
        self.performance_chain = LLMChain(
            llm=self.llm,
            prompt=self.templates.performance_analysis_template,
            verbose=True
        )
    
    def analyze_log_file(self, log_file_path: str) -> Dict[str, Any]:
        """Main analysis function using LangChain"""
        
        # Read and parse log file
        with open(log_file_path, 'r') as file:
            log_content = file.read()
        
        entries = self.parser.parse_log_file(log_content)
        
        # Basic statistics
        total_entries = len(entries)
        error_entries = [e for e in entries if e.status == "ERROR"]
        warning_entries = [e for e in entries if e.status == "WARNING"]
        
        print(f"Parsed {total_entries} log entries")
        print(f"Found {len(error_entries)} errors and {len(warning_entries)} warnings")
        
        # Prepare data for LangChain analysis
        batch_data = self._prepare_batch_data(entries[:50])  # Limit for context
        
        # Run batch analysis
        try:
            batch_result = self.batch_chain.run(
                log_entries=batch_data,
                error_count=len(error_entries),
                total_entries=total_entries
            )
            
            analysis_result = {
                "file_path": log_file_path,
                "total_entries": total_entries,
                "error_count": len(error_entries),
                "warning_count": len(warning_entries),
                "batch_analysis": batch_result,
                "parsed_entries": [asdict(entry) for entry in entries]
            }
            
            # Detailed error analysis if errors found
            if error_entries:
                error_analysis = self._analyze_errors(error_entries, entries)
                analysis_result["error_analysis"] = error_analysis
            
            # Performance analysis
            performance_analysis = self._analyze_performance(entries)
            analysis_result["performance_analysis"] = performance_analysis
            
            return analysis_result
            
        except Exception as e:
            print(f"LangChain analysis failed: {str(e)}")
            return {
                "file_path": log_file_path,
                "total_entries": total_entries,
                "error_count": len(error_entries),
                "warning_count": len(warning_entries),
                "analysis_error": str(e),
                "parsed_entries": [asdict(entry) for entry in entries]
            }
    
    def _prepare_batch_data(self, entries: List[ExtractedLogData]) -> str:
        """Prepare entries for batch analysis"""
        batch_lines = []
        for entry in entries:
            batch_lines.append(f"[{entry.phase}] {entry.component}: {entry.action} - {entry.details[:100]}...")
        return "\n".join(batch_lines)
    
    def _analyze_errors(self, error_entries: List[ExtractedLogData], all_entries: List[ExtractedLogData]) -> str:
        """Analyze errors with surrounding context"""
        error_data = []
        
        for error in error_entries[:10]:  # Limit to first 10 errors
            # Find surrounding context (5 entries before and after)
            error_idx = all_entries.index(error)
            start_idx = max(0, error_idx - 5)
            end_idx = min(len(all_entries), error_idx + 6)
            
            context = all_entries[start_idx:end_idx]
            context_str = "\n".join([f"  {e.details}" for e in context])
            
            error_data.append(f"ERROR: {error.details}\nCONTEXT:\n{context_str}\n")
        
        try:
            return self.error_chain.run(
                error_entries="\n".join(error_data),
                surrounding_context="Boot sequence analysis"
            )
        except Exception as e:
            return f"Error analysis failed: {str(e)}"
    
    def _analyze_performance(self, entries: List[ExtractedLogData]) -> str:
        """Analyze performance metrics"""
        memory_entries = [e for e in entries if e.phase == "MEMORY_MANAGEMENT"]
        driver_entries = [e for e in entries if "LOADING" in e.phase]
        
        memory_data = "\n".join([e.details for e in memory_entries[:20]])
        driver_data = "\n".join([f"{e.component}: {e.details}" for e in driver_entries[:20]])
        
        try:
            return self.performance_chain.run(
                memory_entries=memory_data,
                timing_data="Boot sequence timing analysis",
                driver_load_data=driver_data
            )
        except Exception as e:
            return f"Performance analysis failed: {str(e)}"
    
    def export_results(self, analysis_result: Dict[str, Any], output_dir: str):
        """Export analysis results to multiple formats"""
        import os
        
        base_name = os.path.splitext(os.path.basename(analysis_result["file_path"]))[0]
        
        # Export to JSON
        json_path = os.path.join(output_dir, f"{base_name}_analysis.json")
        with open(json_path, 'w') as f:
            json.dump(analysis_result, f, indent=2)
        
        # Export parsed entries to CSV
        if "parsed_entries" in analysis_result:
            df = pd.DataFrame(analysis_result["parsed_entries"])
            csv_path = os.path.join(output_dir, f"{base_name}_entries.csv")
            df.to_csv(csv_path, index=False)
        
        # Export summary report
        report_path = os.path.join(output_dir, f"{base_name}_report.txt")
        with open(report_path, 'w') as f:
            f.write(self._generate_summary_report(analysis_result))
        
        print(f"Results exported to {output_dir}")
        return {
            "json": json_path,
            "csv": csv_path,
            "report": report_path
        }
    
    def _generate_summary_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate human-readable summary report"""
        report = f"""
            # EDK II Boot Log Analysis Report

            ## File: {analysis_result.get('file_path', 'Unknown')}
            ## Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            ## Summary Statistics
            - Total Entries: {analysis_result.get('total_entries', 0)}
            - Errors Found: {analysis_result.get('error_count', 0)}
            - Warnings Found: {analysis_result.get('warning_count', 0)}

            ## Batch Analysis Results
            {analysis_result.get('batch_analysis', 'No batch analysis available')}
        """
        
        if "error_analysis" in analysis_result:
            report += f"""
                ## Error Analysis
                {analysis_result['error_analysis']}
                """
        
        if "performance_analysis" in analysis_result:
            report += f"""
                ## Performance Analysis
                {analysis_result['performance_analysis']}
            """
        
        return report


def main():
    # Initialize analyzer
    analyzer = EDKIIAnalyzerWithLangChain(model_name="gemma3:latest")
    
    # Analyze the sample log file
    log_file = "boot.log"
    
    try:
        print("Starting EDK II log analysis...")
        results = analyzer.analyze_log_file(log_file)
        
        # Export results
        output_dir = "analysis_output"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = analyzer.export_results(results, output_dir)
        
        print("\nAnalysis Complete!")
        print(f"Results exported to:")
        for file_type, path in exported_files.items():
            print(f"  {file_type.upper()}: {path}")
            
        # Print summary
        print(f"\nQuick Summary:")
        print(f"  Total Entries: {results.get('total_entries', 0)}")
        print(f"  Errors: {results.get('error_count', 0)}")
        print(f"  Warnings: {results.get('warning_count', 0)}")
        
        if results.get('error_count', 0) > 0:
            print(f"  ⚠️  Errors detected - check error analysis report")
        else:
            print(f"  ✅ No errors detected")
            
    except Exception as e:
        print(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
