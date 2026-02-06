"""
Data validation module for quality checks and data integrity.

Provides comprehensive validation including:
- Schema validation
- Data type checks
- Missing value analysis
- Outlier detection
- Temporal consistency checks
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    details: Any = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: datetime
    data_type: str
    row_count: int
    column_count: int
    results: List[ValidationResult] = field(default_factory=list)
    
    @property
    def all_passed(self) -> bool:
        """Check if all validations passed."""
        return all(r.passed for r in self.results)
    
    @property
    def critical_passed(self) -> bool:
        """Check if critical validations passed."""
        critical_checks = ['schema', 'data_types', 'value_ranges']
        return all(
            r.passed for r in self.results 
            if r.name in critical_checks
        )
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Validation Report - {self.timestamp}",
            f"Data type: {self.data_type}",
            f"Shape: {self.row_count} rows × {self.column_count} columns",
            f"Overall: {'PASSED' if self.all_passed else 'FAILED'}",
            "",
            "Results:"
        ]
        
        for result in self.results:
            status = "✓" if result.passed else "✗"
            lines.append(f"  {status} {result.name}: {result.message}")
        
        return "\n".join(lines)


class DataValidator:
    """
    Comprehensive data validation for market data.
    
    Validates:
    - Required columns and schema
    - Data types
    - Missing values
    - Value ranges (positive prices, etc.)
    - Temporal consistency
    - Outliers
    - Data freshness
    """
    
    # Expected columns for different data types
    RAW_COLUMNS = [
        'timestamp', 'symbol', 'open', 'high', 'low', 'close',
        'price', 'volume', 'market_cap'
    ]
    
    PROCESSED_COLUMNS = [
        'timestamp', 'symbol', 'price', 'market_cap'
    ]
    
    # Columns that must be positive
    POSITIVE_COLUMNS = ['open', 'high', 'low', 'close', 'price', 'market_cap']
    
    # Columns that must be non-negative
    NON_NEGATIVE_COLUMNS = ['volume', 'volume_24h']
    
    def __init__(self, config=None):
        """
        Initialize validator.
        
        Args:
            config: Optional configuration object
        """
        self.config = config
    
    def validate_schema(
        self,
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> ValidationResult:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            ValidationResult
        """
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            return ValidationResult(
                name="schema",
                passed=False,
                message=f"Missing columns: {missing}",
                details={'missing_columns': missing}
            )
        
        return ValidationResult(
            name="schema",
            passed=True,
            message="All required columns present"
        )
    
    def validate_data_types(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate data types of columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult
        """
        issues = []
        
        # Check numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'price', 'volume', 'market_cap']
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"{col} is not numeric")
        
        # Check timestamp
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                # Try to convert
                try:
                    pd.to_datetime(df['timestamp'])
                except Exception:
                    issues.append("timestamp cannot be converted to datetime")
        
        if issues:
            return ValidationResult(
                name="data_types",
                passed=False,
                message=f"Type issues: {issues}",
                details={'issues': issues}
            )
        
        return ValidationResult(
            name="data_types",
            passed=True,
            message="All data types valid"
        )
    
    def validate_missing_values(
        self,
        df: pd.DataFrame,
        max_missing_pct: float = 10.0
    ) -> ValidationResult:
        """
        Check for excessive missing values.
        
        Args:
            df: DataFrame to validate
            max_missing_pct: Maximum allowed missing percentage
            
        Returns:
            ValidationResult
        """
        missing_pct = (df.isnull().sum() / len(df)) * 100
        problematic = missing_pct[missing_pct > max_missing_pct]
        
        if len(problematic) > 0:
            return ValidationResult(
                name="missing_values",
                passed=False,
                message=f"High missing % in columns: {problematic.to_dict()}",
                details={'missing_percentages': problematic.to_dict()}
            )
        
        total_missing = df.isnull().sum().sum()
        return ValidationResult(
            name="missing_values",
            passed=True,
            message=f"Total missing values: {total_missing}"
        )
    
    def validate_value_ranges(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate that values are within expected ranges.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult
        """
        issues = []
        
        # Check positive columns
        for col in self.POSITIVE_COLUMNS:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    issues.append(f"{col}: {negative_count} non-positive values")
        
        # Check non-negative columns
        for col in self.NON_NEGATIVE_COLUMNS:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"{col}: {negative_count} negative values")
        
        if issues:
            return ValidationResult(
                name="value_ranges",
                passed=False,
                message=f"Range issues: {issues}",
                details={'issues': issues}
            )
        
        return ValidationResult(
            name="value_ranges",
            passed=True,
            message="All values within expected ranges"
        )
    
    def validate_temporal_consistency(
        self,
        df: pd.DataFrame,
        max_gap_days: int = 7
    ) -> ValidationResult:
        """
        Validate temporal ordering and gaps.
        
        Args:
            df: DataFrame to validate
            max_gap_days: Maximum allowed gap between data points
            
        Returns:
            ValidationResult
        """
        if 'timestamp' not in df.columns:
            return ValidationResult(
                name="temporal_consistency",
                passed=False,
                message="No timestamp column found"
            )
        
        issues = []
        
        # Check sorting
        if not df['timestamp'].is_monotonic_increasing:
            issues.append("Data not sorted chronologically")
        
        # Check duplicates
        duplicates = df['timestamp'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate timestamps")
        
        # Check gaps
        if len(df) > 1:
            time_diffs = df['timestamp'].diff()
            max_gap = time_diffs.max()
            
            if max_gap > timedelta(days=max_gap_days):
                issues.append(f"Large gap detected: {max_gap.days} days")
        
        if issues:
            return ValidationResult(
                name="temporal_consistency",
                passed=False,
                message=f"Temporal issues: {issues}",
                details={'issues': issues}
            )
        
        return ValidationResult(
            name="temporal_consistency",
            passed=True,
            message="Data temporally consistent"
        )
    
    def validate_outliers(
        self,
        df: pd.DataFrame,
        z_threshold: float = 5.0
    ) -> ValidationResult:
        """
        Detect statistical outliers using z-score.
        
        Args:
            df: DataFrame to validate
            z_threshold: Z-score threshold for outliers
            
        Returns:
            ValidationResult (informational, always passes)
        """
        outlier_counts = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['day_of_week', 'month', 'day_of_month', 'quarter']
        
        for col in numeric_cols:
            if col not in exclude_cols and df[col].std() > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > z_threshold).sum()
                
                if outliers > 0:
                    outlier_counts[col] = int(outliers)
        
        if outlier_counts:
            return ValidationResult(
                name="outliers",
                passed=True,  # Informational only
                message=f"Outliers detected (z>{z_threshold}): {outlier_counts}",
                details={'outlier_counts': outlier_counts}
            )
        
        return ValidationResult(
            name="outliers",
            passed=True,
            message="No significant outliers detected"
        )
    
    def validate_data_freshness(
        self,
        df: pd.DataFrame,
        max_age_hours: float = 48
    ) -> ValidationResult:
        """
        Check if data is recent enough.
        
        Args:
            df: DataFrame to validate
            max_age_hours: Maximum allowed data age in hours
            
        Returns:
            ValidationResult
        """
        if 'timestamp' not in df.columns:
            return ValidationResult(
                name="data_freshness",
                passed=False,
                message="No timestamp column"
            )
        
        latest = df['timestamp'].max()
        
        # Handle timezone-naive comparison
        now = datetime.utcnow()
        if latest.tzinfo is not None:
            from datetime import timezone
            now = datetime.now(timezone.utc)
        
        age_hours = (now - latest).total_seconds() / 3600
        
        if age_hours > max_age_hours:
            return ValidationResult(
                name="data_freshness",
                passed=False,
                message=f"Data is {age_hours:.1f} hours old (threshold: {max_age_hours}h)",
                details={'age_hours': age_hours}
            )
        
        return ValidationResult(
            name="data_freshness",
            passed=True,
            message=f"Data age: {age_hours:.1f} hours"
        )
    
    def validate(
        self,
        df: pd.DataFrame,
        data_type: str = "raw"
    ) -> ValidationReport:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data ('raw' or 'processed')
            
        Returns:
            ValidationReport with all results
        """
        logger.info(f"Running validation for {data_type} data...")
        
        # Determine required columns
        required_columns = (
            self.RAW_COLUMNS if data_type == "raw" 
            else self.PROCESSED_COLUMNS
        )
        
        # Create report
        report = ValidationReport(
            timestamp=datetime.now(),
            data_type=data_type,
            row_count=len(df),
            column_count=len(df.columns)
        )
        
        # Run validations
        validations = [
            ('schema', lambda: self.validate_schema(df, required_columns)),
            ('data_types', lambda: self.validate_data_types(df)),
            ('missing_values', lambda: self.validate_missing_values(df)),
            ('value_ranges', lambda: self.validate_value_ranges(df)),
            ('temporal_consistency', lambda: self.validate_temporal_consistency(df)),
            ('outliers', lambda: self.validate_outliers(df)),
            ('data_freshness', lambda: self.validate_data_freshness(df))
        ]
        
        for name, validation_func in validations:
            try:
                result = validation_func()
                report.results.append(result)
                
                log_level = logging.INFO if result.passed else logging.WARNING
                logger.log(log_level, f"  {name}: {result.message}")
                
            except Exception as e:
                report.results.append(ValidationResult(
                    name=name,
                    passed=False,
                    message=f"Error: {str(e)}"
                ))
                logger.error(f"  {name}: Error - {e}")
        
        # Summary
        status = "PASSED" if report.all_passed else "FAILED"
        logger.info(f"Validation {status} ({len([r for r in report.results if r.passed])}/{len(report.results)} checks)")
        
        return report
