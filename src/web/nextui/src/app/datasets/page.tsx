import React, { Suspense } from 'react';
import Datasets from './Datasets';

export default function Page() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <div>
        <Datasets />
      </div>
    </Suspense>
  );
}
