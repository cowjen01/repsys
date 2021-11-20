import { useState, useEffect } from 'react';

export function fetchPredictions(path, body = {}) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState();
  const [isLoading, setIsLoading] = useState(true);

  const encodedBody = JSON.stringify(body);

  useEffect(() => {
    let isActive = true;

    setIsLoading(true);

    fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: encodedBody,
    })
      .then((response) => response.json())
      .then((data) => {
        if (isActive) {
          setItems(data);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        setIsLoading(false);
        setError(err);
      });

    return () => {
      isActive = false;
    };
  }, [encodedBody]);

  return { items, isLoading, error };
}
