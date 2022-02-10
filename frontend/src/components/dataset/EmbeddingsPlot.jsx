import React, { useState, useMemo, useRef } from 'react';
import { Paper, Stack, Backdrop, Box, CircularProgress } from '@mui/material';
import Plotly from 'plotly.js';

import { ScatterPlot } from '../plots';
import { CategoryFilter } from '../filters';
import { plotColors } from '../../const';

const itemEmbeddings = [
  {
    x: 0.72,
    y: 0.7,
    c: 3,
    label: 'Ghosts of Mississippi',
    year: 1968,
    id: 1307,
    country: 'CO',
    genres: ['comedy', 'action'],
  },
  {
    x: 0.37,
    y: 0.78,
    c: 1,
    id: 1303,
    label: 'American Pie 2',
    year: 1996,
    country: 'CO',
    genres: ['action'],
  },
  {
    x: 0.07,
    y: 0.44,
    c: 5,
    id: 1302,
    label: 'Inglorious Bastards (Quel maledetto treno blindato)',
    year: 1997,
    country: 'MK',
    genres: ['comedy', 'horror', 'action'],
  },
  {
    x: 0.07,
    y: 0.93,
    c: 3,
    id: 1300,
    label: 'Prince of Egypt, The',
    year: 2001,
    country: 'FR',
    genres: ['action'],
  },
  {
    x: 0.63,
    y: 0.79,
    c: 3,
    id: 1322,
    label: 'Friday the 13th',
    year: 2010,
    country: 'CN',
    genres: ['horror'],
  },
  {
    x: 0.65,
    y: 0.4,
    c: 2,
    id: 1325,
    label: 'Pearl Harbor',
    year: 2015,
    country: 'ID',
    genres: ['drama'],
  },
  {
    x: 0.42,
    y: 0.79,
    c: 3,
    id: 1329,
    label: 'Ghosts of Mississippi',
    year: 1968,
    country: 'CO',
    genres: ['comedy', 'action'],
  },
  {
    x: 0.17,
    y: 0.98,
    c: 1,
    id: 13212,
    label: 'American Pie 2',
    year: 1996,
    country: 'CO',
    genres: ['action'],
  },
  {
    x: 0.37,
    y: 0.84,
    c: 5,
    id: 13232,
    label: 'Inglorious Bastards (Quel maledetto treno blindato)',
    year: 1997,
    country: 'MK',
    genres: ['comedy', 'horror', 'action'],
  },
  {
    x: 0.27,
    y: 0.43,
    c: 3,
    id: 13543,
    label: 'Prince of Egypt, The',
    year: 2001,
    country: 'FR',
    genres: ['action'],
  },
  {
    x: 0.43,
    y: 0.29,
    c: 3,
    id: 1312,
    label: 'Friday the 13th',
    year: 2010,
    country: 'CN',
    genres: ['horror'],
  },
  {
    x: 0.6,
    y: 0.44,
    c: 2,
    id: 13765,
    label: 'Pearl Harbor',
    year: 2015,
    country: 'ID',
    genres: ['drama'],
  },
];

function getItemIds(argumentName, argumentValue) {
  return [13765, 1312, 1325, 1322];
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function EmbeddingsPlot({ columns, onSelect }) {
  const [isLoading, setIsLoading] = useState(false);
  const [selectedField, setSelectedField] = useState('');
  const [selectedValues, setSelectedValues] = useState([]);
  const [isPlotSelection, setIsPlotSelection] = useState(false);
  const [highlightedPoints, setHighlightedPoints] = useState([]);
  const scatterRef = useRef();

  const scatterColors = useMemo(() => {
    if (!highlightedPoints.length) {
      return plotColors.selectedMarker;
    }

    const colors = [];
    for (let i = 0; i < itemEmbeddings.length; i += 1) {
      colors.push(plotColors.unselectedMarker);
    }

    highlightedPoints.forEach((p) => {
      colors[p] = plotColors.selectedMarker;
    });

    return colors;
  }, [highlightedPoints]);

  const filterOptions = useMemo(() => {
    if (selectedField) {
      const col = columns[selectedField];
      if (col.dtype === 'number') {
        return col.bins.map((bin, index) => ({
          value: index,
          label: `${bin[0]} - ${bin[1]}`,
        }));
      }
      return col.options.map((option) => ({
        value: option,
        label: option,
      }));
    }
    return [];
  }, [selectedField]);

  const scatterPoints = useMemo(
    () => ({
      x: itemEmbeddings.map(({ x }) => x),
      y: itemEmbeddings.map(({ y }) => y),
      meta: itemEmbeddings.map(({ id }) => ({ id })),
      label: itemEmbeddings.map(({ label }) => label),
    }),
    []
  );

  const isMultipleSelect = useMemo(() => {
    if (selectedField) {
      const fieldType = columns[selectedField].dtype;
      return fieldType === 'tags' || fieldType === 'category';
    }
    return false;
  }, [selectedField]);

  const resetFilterSelection = () => {
    setSelectedValues([]);
    setSelectedField('');
  };

  const resetScatterSelection = () => {
    setIsPlotSelection(false);
    Plotly.restyle(scatterRef.current.el, { selectedpoints: [null] });
  };

  const handleFilterApply = async () => {
    setIsLoading(true);
    await sleep(500);
    setIsLoading(false);

    const ids = getItemIds(selectedField, selectedValues);

    if (!ids.length) {
      setHighlightedPoints([]);
    } else {
      const indices = itemEmbeddings.reduce((acc, item, index) => {
        if (ids.includes(item.id)) {
          acc.push(index);
        }
        return acc;
      }, []);
      setHighlightedPoints(indices);
    }
  };

  const handleAttributeChange = (newValue) => {
    setSelectedField(newValue);
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
      const itemsIds = points.map((p) => p.customdata.id);
      onSelect(itemsIds);
      setIsPlotSelection(true);
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

  return (
    <Paper sx={{ p: 2 }}>
      <Stack direction="row" spacing={2}>
        <CategoryFilter
          label="Item attribute"
          displayEmpty
          value={selectedField}
          onChange={handleAttributeChange}
          options={Object.keys(columns).map((col) => ({
            value: col,
            label: col,
          }))}
        />
        <CategoryFilter
          label="Attribute value"
          disabled={!selectedField}
          value={isMultipleSelect ? selectedValues : selectedValues[0]}
          multiple={isMultipleSelect}
          onClose={handleFilterApply}
          onChange={handleValuesChange}
          options={filterOptions}
        />
      </Stack>
      <Box position="relative">
        {isLoading && (
          <Backdrop
            sx={{
              position: 'absolute',
              color: '#000',
              backgroundColor: 'rgba(255,255,255,0.6)',
              zIndex: 100,
            }}
            open
          >
            <CircularProgress color="inherit" />
          </Backdrop>
        )}

        <ScatterPlot
          height={450}
          x={scatterPoints.x}
          y={scatterPoints.y}
          meta={scatterPoints.meta}
          color={scatterColors}
          label={scatterPoints.label}
          innerRef={scatterRef}
          onDeselect={handleScatterUnselect}
          onSelected={handleScatterSelect}
        />
      </Box>
    </Paper>
  );
}

export default EmbeddingsPlot;
