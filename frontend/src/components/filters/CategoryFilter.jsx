import React from 'react';
import pt from 'prop-types';
import { FormControl, InputLabel, Select, MenuItem } from '@mui/material';

function CategoryFilter({ onChange, onClose, value, options, label, displayEmpty, ...rest }) {
  const handleChange = (e) => {
    onChange(e.target.value);
  };

  const handleClose = () => {
    setTimeout(() => {
      document.activeElement.blur();
    }, 0);
    onClose();
  };

  return (
    <FormControl variant="filled" sx={{ minWidth: 250 }}>
      <InputLabel>{label}</InputLabel>
      <Select value={value} label={label} onChange={handleChange} onClose={handleClose} {...rest}>
        {displayEmpty && (
          <MenuItem value="">
            <em>None</em>
          </MenuItem>
        )}
        {options.map((option) => (
          <MenuItem key={option.value} value={option.value}>
            {option.label}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}

CategoryFilter.defaultProps = {
  value: '',
  displayEmpty: false,
  onClose: () => {},
};

CategoryFilter.propTypes = {
  onChange: pt.func.isRequired,
  onClose: pt.func,
  label: pt.string.isRequired,
  value: pt.oneOfType([pt.string, pt.number, pt.arrayOf(pt.string)]),
  displayEmpty: pt.bool,
  options: pt.arrayOf(
    pt.shape({
      value: pt.oneOfType([pt.string, pt.number]),
      label: pt.string,
    })
  ).isRequired,
};

export default CategoryFilter;
