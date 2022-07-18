import React from 'react';
import pt from 'prop-types';
import { FormControl, InputLabel, Select, MenuItem } from '@mui/material';

function CategoryFilter({ onChange, value, options, label, displayEmpty, width, ...rest }) {
  const handleChange = (e) => {
    onChange(e.target.value);
  };

  const handleClose = () => {
    setTimeout(() => {
      document.activeElement.blur();
    }, 100);
  };

  return (
    <FormControl size="small" variant="filled" sx={{ width }}>
      <InputLabel>{label}</InputLabel>
      <Select value={value} label={label} onChange={handleChange} onClose={handleClose} {...rest}>
        {displayEmpty && (
          <MenuItem value="">
            <em>None</em>
          </MenuItem>
        )}
        {options.map((option) => {
          if (
            typeof option === 'string' ||
            option instanceof String ||
            typeof option === 'number'
          ) {
            return (
              <MenuItem key={option} value={option}>
                {option}
              </MenuItem>
            );
          }
          return (
            <MenuItem key={option.value} value={option.value}>
              {option.label}
            </MenuItem>
          );
        })}
      </Select>
    </FormControl>
  );
}

CategoryFilter.defaultProps = {
  value: '',
  displayEmpty: false,
  width: 170,
};

CategoryFilter.propTypes = {
  onChange: pt.func.isRequired,
  label: pt.string.isRequired,
  width: pt.number,
  value: pt.oneOfType([pt.string, pt.number, pt.arrayOf(pt.string)]),
  displayEmpty: pt.bool,
  options: pt.oneOfType([
    pt.arrayOf(pt.oneOfType([pt.string, pt.number])),
    pt.arrayOf(
      pt.shape({
        value: pt.oneOfType([pt.string, pt.number]),
        label: pt.string,
      })
    ),
  ]).isRequired,
};

export default CategoryFilter;
