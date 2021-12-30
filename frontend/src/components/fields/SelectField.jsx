/* eslint-disable react/prop-types */
import React from 'react';
import TextField from '@mui/material/TextField';

function SelectField({
  field: { name, value, onChange, onBlur },
  form: { errors },
  options,
  displayEmpty,
  ...props
}) {
  return (
    <TextField
      select
      value={value}
      onChange={onChange}
      onBlur={onBlur}
      name={name}
      fullWidth
      helperText={errors[name]}
      error={Boolean(errors[name])}
      variant="outlined"
      margin="normal"
      SelectProps={{
        native: true,
      }}
      {...props}
    >
      {displayEmpty && <option value="">{null}</option>}
      {options.map((option) => (
        <option key={option.value || 'empty'} value={option.value}>
          {option.label}
        </option>
      ))}
    </TextField>
  );
}

export default SelectField;
