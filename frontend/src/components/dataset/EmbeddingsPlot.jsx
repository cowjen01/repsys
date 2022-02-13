import React, { useState, useMemo, useRef, useEffect } from 'react';
import pt from 'prop-types';
import { Paper, Stack, TextField } from '@mui/material';
import Plotly from 'plotly.js';

import { ScatterPlot } from '../plots';
import { CategoryFilter } from '../filters';
import { plotColors } from '../../const';
import { capitalize } from '../../utils';

function EmbeddingsPlot({
  attributes,
  onSelect,
  onFilterApply,
  displayThreshold,
  filterResults,
  embeddings,
}) {
  const [selectedAttribute, setSelectedAttribute] = useState('');
  const [selectedValues, setSelectedValues] = useState([]);
  const [selectedThreshold, setSelectedThreshold] = useState(5);
  const [highlightedPoints, setHighlightedPoints] = useState([]);
  const scatterRef = useRef();

  useEffect(() => {
    if (filterResults.length) {
      const indices = embeddings.reduce((acc, item, index) => {
        if (filterResults.includes(item.id)) {
          acc.push(index);
        }
        return acc;
      }, []);
      setHighlightedPoints(indices);
    }
  }, [filterResults]);

  const scatterColors = useMemo(() => {
    if (!highlightedPoints.length) {
      return plotColors.selectedMarker;
    }

    const colors = [];
    for (let i = 0; i < embeddings.length; i += 1) {
      colors.push(plotColors.unselectedMarker);
    }

    highlightedPoints.forEach((p) => {
      colors[p] = plotColors.selectedMarker;
    });

    return colors;
  }, [highlightedPoints]);

  const attributeOptions = useMemo(
    () =>
      Object.entries(attributes)
        .filter((attribute) => ['number', 'category', 'tags'].includes(attribute[1].dtype))
        .map((attribute) => ({
          value: attribute[0],
          label: capitalize(attribute[0]),
        })),
    [attributes]
  );

  const filterOptions = useMemo(() => {
    if (selectedAttribute) {
      const col = attributes[selectedAttribute];
      if (col.dtype === 'number') {
        return col.bins.slice(1).map((bin, index) => ({
          value: index,
          label: `${col.bins[index]} - ${bin}`,
        }));
      }
      return col.options.map((option) => ({
        value: option,
        label: option,
      }));
    }
    return [];
  }, [selectedAttribute]);

  const scatterPoints = useMemo(
    () => ({
      x: embeddings.map(({ x }) => x),
      y: embeddings.map(({ y }) => y),
      meta: embeddings.map(({ id }) => ({ id })),
      label: embeddings.map(({ label }) => label),
    }),
    [embeddings]
  );

  const isMultipleSelect = useMemo(() => {
    if (selectedAttribute) {
      const fieldType = attributes[selectedAttribute].dtype;
      return fieldType === 'tags';
    }
    return false;
  }, [selectedAttribute]);

  const resetFilterSelection = () => {
    setSelectedValues([]);
    setSelectedAttribute('');
  };

  const resetScatterSelection = () => {
    Plotly.restyle(scatterRef.current.el, { selectedpoints: [null] });
  };

  const handleFilterApply = () => {
    if (selectedValues.length) {
      const fieldType = attributes[selectedAttribute].dtype;
      if (fieldType === 'number') {
        const { bins } = attributes[selectedAttribute];
        const index = selectedValues[0];
        onFilterApply({
          attribute: selectedAttribute,
          range: [bins[index], bins[index + 1]],
          threshold: selectedThreshold,
        });
      } else {
        onFilterApply({
          attribute: selectedAttribute,
          values: selectedValues,
          threshold: parseInt(selectedThreshold, 10),
        });
      }
    }
  };

  const handleAttributeChange = (newValue) => {
    setSelectedAttribute(newValue);
    setSelectedValues([]);
    setHighlightedPoints([]);
    resetScatterSelection();
    onSelect([]);
  };

  const handleValuesChange = (newValue) => {
    if (isMultipleSelect) {
      setSelectedValues(newValue);
    } else {
      setSelectedValues([newValue]);
    }
  };

  const handleScatterSelect = (eventData) => {
    if (eventData && eventData.points.length) {
      const { points } = eventData;
      onSelect(points.map((p) => p.customdata.id));
      setHighlightedPoints(points[0].data.selectedpoints);
      resetFilterSelection();
    }
  };

  const handleScatterUnselect = () => {
    onSelect([]);
    setHighlightedPoints([]);
    resetScatterSelection();
    resetFilterSelection();
  };

  const handleThresholdChange = (event) => {
    setSelectedThreshold(event.target.value);
  };

  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Stack direction="row" spacing={2}>
        <CategoryFilter
          label="Item attribute"
          displayEmpty
          value={selectedAttribute}
          onChange={handleAttributeChange}
          options={attributeOptions}
        />
        <CategoryFilter
          label="Attribute value"
          disabled={!selectedAttribute}
          value={isMultipleSelect ? selectedValues : selectedValues[0]}
          multiple={isMultipleSelect}
          onBlur={handleFilterApply}
          onChange={handleValuesChange}
          options={filterOptions}
        />
        {displayThreshold && (
          <TextField
            sx={{ minWidth: 250 }}
            disabled={!selectedAttribute}
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
      <ScatterPlot
        height={450}
        // isLoading={isLoading}
        x={scatterPoints.x}
        y={scatterPoints.y}
        meta={scatterPoints.meta}
        color={scatterColors}
        label={scatterPoints.label}
        innerRef={scatterRef}
        onDeselect={handleScatterUnselect}
        onSelected={handleScatterSelect}
      />
    </Paper>
  );
}

EmbeddingsPlot.defaultProps = {
  displayThreshold: false,
  filterResults: [],
  embeddings: [],
};

EmbeddingsPlot.propTypes = {
  onSelect: pt.func.isRequired,
  onFilterApply: pt.func.isRequired,
  displayThreshold: pt.bool,
  // eslint-disable-next-line react/forbid-prop-types
  attributes: pt.any.isRequired,
  filterResults: pt.arrayOf(pt.number),
  embeddings: pt.arrayOf(
    pt.shape({
      title: pt.string,
      x: pt.number,
      y: pt.number,
      id: pt.number,
    })
  ),
};

export default EmbeddingsPlot;
