import React, { useState, useMemo, useEffect } from 'react';
import pt from 'prop-types';
import { Stack, TextField } from '@mui/material';

import { CategoryFilter } from '../filters';
import { capitalize } from '../../utils';

function AttributesSelector({
  attributes,
  displayThreshold,
  onFilterApply,
  disabled,
  resetIndex,
  onChange,
}) {
  const [selectedAttribute, setSelectedAttribute] = useState('');
  const [selectedValue, setSelectedValue] = useState([]);
  const [selectedThreshold, setSelectedThreshold] = useState(1);

  // const isMultipleSelect = !!(selectedAttribute && attributes[selectedAttribute].dtype === 'tag');

  const attributeOptions = useMemo(() => {
    const allowedTypes = ['number', 'category', 'tag'];
    return Object.entries(attributes)
      .filter((attribute) => allowedTypes.includes(attribute[1].dtype))
      .map((attribute) => ({
        value: attribute[0],
        label: capitalize(attribute[0]),
      }));
  }, [attributes]);

  const filterOptions = useMemo(() => {
    if (selectedAttribute) {
      const col = attributes[selectedAttribute];
      if (col.dtype === 'number') {
        return col.bins.slice(1).map((bin, index) => ({
          value: index,
          label: `${col.bins[index]} - ${bin}`,
        }));
      }
      return col.options;
    }
    return [];
  }, [selectedAttribute]);

  useEffect(() => {
    if (resetIndex > 0) {
      setSelectedAttribute('');
      setSelectedValue(null);
    }
  }, [resetIndex]);

  const handleThresholdChange = (event) => {
    setSelectedThreshold(event.target.value);
  };

  const handleAttributeChange = (newValue) => {
    setSelectedAttribute(newValue);
    setSelectedValue(null);
    onChange();
  };

  const handleValuesChange = (newValue) => {
    // if (isMultipleSelect) {
    //   setSelectedValues(newValue);
    // } else {
    //   setSelectedValues([newValue]);
    // }
    setSelectedValue(newValue);
  };

  const handleFilterApply = () => {
    if (selectedValue) {
      const fieldType = attributes[selectedAttribute].dtype;
      const query = {
        attribute: selectedAttribute,
      };
      if (fieldType === 'number') {
        const { bins } = attributes[selectedAttribute];
        query.range = [bins[selectedValue], bins[selectedValue + 1]];
      } else {
        query.values = [selectedValue];
      }
      if (displayThreshold) {
        query.threshold = parseInt(selectedThreshold, 10);
      }
      onFilterApply(query);
    }
  };

  return (
    <Stack direction="row" spacing={1}>
      <CategoryFilter
        label="Item attribute"
        displayEmpty
        disabled={disabled}
        value={selectedAttribute}
        onChange={handleAttributeChange}
        options={attributeOptions}
      />
      <CategoryFilter
        label="Attribute value"
        disabled={!selectedAttribute || disabled}
        value={selectedValue}
        onBlur={handleFilterApply}
        onChange={handleValuesChange}
        options={filterOptions}
      />
      {displayThreshold && (
        <TextField
          sx={{ width: 200 }}
          disabled={!selectedAttribute || disabled}
          label="Min. interactions"
          type="number"
          onBlur={handleFilterApply}
          inputProps={{ inputMode: 'numeric', pattern: '[0-9]*', min: 1 }}
          onChange={handleThresholdChange}
          value={selectedThreshold}
          variant="filled"
        />
      )}
    </Stack>
  );
}

AttributesSelector.defaultProps = {
  displayThreshold: false,
  resetIndex: 0,
  disabled: false,
};

AttributesSelector.propTypes = {
  onFilterApply: pt.func.isRequired,
  onChange: pt.func.isRequired,
  displayThreshold: pt.bool,
  disabled: pt.bool,
  resetIndex: pt.number,
  // eslint-disable-next-line react/forbid-prop-types
  attributes: pt.any.isRequired,
};

export default AttributesSelector;
